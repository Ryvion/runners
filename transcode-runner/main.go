package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
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
	writeJSON("/work/metrics.json", map[string]any{
		"engine":       "ffmpeg",
		"duration_ms":  durationMs,
		"output_bytes": size,
	})
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
