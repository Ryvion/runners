package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type JobSpec struct {
	PayloadURL string   `json:"payload_url"`
	InputURL   string   `json:"input_url"`
	InputFile  string   `json:"input_file"`
	OutputFile string   `json:"output_file"`
	Args       []string `json:"args"`
}

type receipt struct {
	OutputHash string `json:"output_hash"`
	OK         bool   `json:"ok"`
}

type ffprobeOutput struct {
	Streams []ffprobeStream `json:"streams"`
	Format  ffprobeFormat   `json:"format"`
}

type ffprobeStream struct {
	CodecType    string `json:"codec_type"`
	CodecName    string `json:"codec_name"`
	Width        int    `json:"width"`
	Height       int    `json:"height"`
	AvgFrameRate string `json:"avg_frame_rate"`
	Channels     int    `json:"channels"`
	SampleRate   string `json:"sample_rate"`
}

type ffprobeFormat struct {
	FormatName string `json:"format_name"`
	Duration   string `json:"duration"`
	Size       string `json:"size"`
	BitRate    string `json:"bit_rate"`
}

func main() {
	js := loadJob()
	inputPath, outputPath := resolvePaths(js)
	prepareInput(js, inputPath)

	args := buildArgs(js, inputPath, outputPath)
	start := time.Now()
	cmd := exec.Command("ffmpeg", args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		log.Fatalf("ffmpeg: %v\n%s", err, string(out))
	}
	durationMs := time.Since(start).Milliseconds()

	sum, size := hashFile(outputPath)
	writeJSON("/work/receipt.json", receipt{OutputHash: sum, OK: true})
	metrics := map[string]any{
		"engine":       "ffmpeg",
		"duration_ms":  durationMs,
		"output_bytes": size,
		"output_name":  filepath.Base(outputPath),
	}
	if ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(outputPath)), "."); ext != "" {
		metrics["output_format"] = ext
	}
	for key, value := range probeOutput(outputPath) {
		metrics[key] = value
	}
	writeJSON("/work/metrics.json", metrics)
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"output_hash": sum,
		"ok":          true,
	})
}

func loadJob() JobSpec {
	f, err := os.Open("/work/job.json")
	if err != nil {
		log.Fatalf("open job spec: %v", err)
	}
	defer f.Close()

	var js JobSpec
	if err := json.NewDecoder(f).Decode(&js); err != nil {
		log.Fatalf("decode job spec: %v", err)
	}
	return js
}

func resolvePaths(js JobSpec) (string, string) {
	inputPath := strings.TrimSpace(js.InputFile)
	if inputPath == "" {
		inputPath = "/work/input"
	}
	outputPath := strings.TrimSpace(js.OutputFile)
	if outputPath == "" {
		outputPath = "/work/output"
	}
	return inputPath, outputPath
}

func prepareInput(js JobSpec, inputPath string) {
	src := strings.TrimSpace(js.PayloadURL)
	if src == "" {
		src = strings.TrimSpace(js.InputURL)
	}
	if src == "" {
		if _, err := os.Stat(inputPath); err != nil {
			log.Fatalf("no payload provided and input file missing: %v", err)
		}
		return
	}
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Get(src)
	if err != nil {
		log.Fatalf("download payload: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		log.Fatalf("download payload: unexpected status %s", resp.Status)
	}
	out, err := os.Create(inputPath)
	if err != nil {
		log.Fatalf("create input file: %v", err)
	}
	defer out.Close()
	if _, err := io.Copy(out, resp.Body); err != nil {
		log.Fatalf("copy payload: %v", err)
	}
}

func buildArgs(js JobSpec, inputPath, outputPath string) []string {
	if len(js.Args) == 0 {
		return []string{
			"-y",
			"-i", inputPath,
			"-c:v", "libx264",
			"-preset", "veryfast",
			"-movflags", "+faststart",
			"-pix_fmt", "yuv420p",
			"-c:a", "aac",
			"-b:a", "128k",
			"-f", "mp4",
			outputPath,
		}
	}
	args := make([]string, 0, len(js.Args))
	for _, a := range js.Args {
		a = strings.TrimSpace(a)
		switch a {
		case "{input}":
			a = inputPath
		case "{output}":
			a = outputPath
		}
		args = append(args, a)
	}
	return args
}

func hashFile(path string) (string, int64) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open output: %v", err)
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		log.Fatalf("hash output: %v", err)
	}
	return hex.EncodeToString(h.Sum(nil)), n
}

func probeOutput(path string) map[string]any {
	cmd := exec.Command(
		"ffprobe",
		"-v", "error",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		path,
	)
	out, err := cmd.Output()
	if err != nil {
		return map[string]any{
			"probe_error": err.Error(),
		}
	}

	var probe ffprobeOutput
	if err := json.Unmarshal(out, &probe); err != nil {
		return map[string]any{
			"probe_error": fmt.Sprintf("decode ffprobe: %v", err),
		}
	}

	metrics := map[string]any{}
	if name := strings.TrimSpace(probe.Format.FormatName); name != "" {
		metrics["container_family"] = firstCSVToken(name)
	}
	if seconds, ok := parseFloat(probe.Format.Duration); ok && seconds > 0 {
		metrics["output_duration_seconds"] = seconds
	}
	if bitrate, ok := parseInt(probe.Format.BitRate); ok && bitrate > 0 {
		metrics["output_bitrate_bps"] = bitrate
	}
	for _, stream := range probe.Streams {
		switch strings.TrimSpace(stream.CodecType) {
		case "video":
			if codec := strings.TrimSpace(stream.CodecName); codec != "" {
				metrics["video_codec"] = codec
			}
			if stream.Width > 0 {
				metrics["width"] = stream.Width
			}
			if stream.Height > 0 {
				metrics["height"] = stream.Height
			}
			if fps, ok := parseFrameRate(stream.AvgFrameRate); ok && fps > 0 {
				metrics["frame_rate"] = fps
			}
		case "audio":
			if codec := strings.TrimSpace(stream.CodecName); codec != "" {
				metrics["audio_codec"] = codec
			}
			if stream.Channels > 0 {
				metrics["audio_channels"] = stream.Channels
			}
			if sampleRate, ok := parseInt(stream.SampleRate); ok && sampleRate > 0 {
				metrics["audio_sample_rate_hz"] = sampleRate
			}
		}
	}
	return metrics
}

func firstCSVToken(value string) string {
	if idx := strings.Index(value, ","); idx >= 0 {
		return strings.TrimSpace(value[:idx])
	}
	return strings.TrimSpace(value)
}

func parseFloat(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, false
	}
	return parsed, true
}

func parseInt(value string) (int64, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0, false
	}
	return parsed, true
}

func parseFrameRate(value string) (float64, bool) {
	value = strings.TrimSpace(value)
	if value == "" || value == "0/0" {
		return 0, false
	}
	if strings.Contains(value, "/") {
		parts := strings.SplitN(value, "/", 2)
		num, okNum := parseFloat(parts[0])
		den, okDen := parseFloat(parts[1])
		if !okNum || !okDen || den == 0 {
			return 0, false
		}
		return num / den, true
	}
	return parseFloat(value)
}

func writeJSON(path string, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		log.Fatalf("marshal %s: %v", path, err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		log.Fatalf("write %s: %v", path, err)
	}
}

func init() {
	log.SetFlags(0)
	log.SetPrefix("transcode-runner: ")
}
