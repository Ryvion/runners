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
	"path/filepath"
	"strings"
	"time"
)

var allowedCmds = map[string]bool{
	"llama-embed":    true,
	"llama-cli":      true,
	"llama-quantize": true,
}

type JobSpec struct {
	PayloadURL string   `json:"payload_url"`
	Text       string   `json:"text"`
	InputFile  string   `json:"input_file"`
	OutputFile string   `json:"output_file"`
	Cmd        string   `json:"cmd"`
	Args       []string `json:"args"`
}

type receipt struct {
	OutputHash string `json:"output_hash"`
	OK         bool   `json:"ok"`
	Model      string `json:"model,omitempty"`
	Dimensions int    `json:"dimensions,omitempty"`
}

func main() {
	js := loadJob()
	inputPath, outputPath := resolvePaths(&js)
	prepareInput(js, inputPath)

	cmdBase, args := buildCommand(js, inputPath, outputPath)
	if !allowedCmds[cmdBase] {
		log.Fatalf("blocked command: %q", cmdBase)
	}

	cmd := exec.Command(cmdBase, args...)
	cmd.Env = append(os.Environ(),
		"RYV_EMBED_INPUT="+inputPath,
		"RYV_EMBED_OUTPUT="+outputPath,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		log.Fatalf("embed: %v\n%s", err, string(out))
	}

	sum, size := hashFile(outputPath)
	writeReceipt(receipt{
		OutputHash: sum,
		OK:         true,
	})
	writeMetrics(map[string]any{
		"engine":       "embedding",
		"output_bytes": size,
	})
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"output_hash": sum,
		"ok":          true,
		"output_path": outputPath,
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

func resolvePaths(js *JobSpec) (string, string) {
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
	switch {
	case strings.TrimSpace(js.Text) != "":
		if err := os.WriteFile(inputPath, []byte(js.Text), 0o644); err != nil {
			log.Fatalf("write inline text payload: %v", err)
		}
	case strings.TrimSpace(js.PayloadURL) != "":
		downloadToFile(js.PayloadURL, inputPath)
	default:
		if _, err := os.Stat(inputPath); err != nil {
			log.Fatalf("no payload provided and input file missing: %v", err)
		}
	}
}

func buildCommand(js JobSpec, inputPath, outputPath string) (string, []string) {
	rawCmd := strings.TrimSpace(js.Cmd)
	cmdBase := ""
	if rawCmd != "" {
		cmdBase = filepath.Base(rawCmd)
	}
	if cmdBase == "" || cmdBase == "." {
		cmdBase = "llama-embed"
	}
	args := make([]string, 0, len(js.Args)+4)
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
	if len(args) == 0 && cmdBase == "llama-embed" {
		args = []string{"--input", inputPath, "--output", outputPath}
	}
	return cmdBase, args
}

func downloadToFile(srcURL, dst string) {
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(srcURL)
	if err != nil {
		log.Fatalf("download payload: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		log.Fatalf("download payload: unexpected status %s", resp.Status)
	}
	out, err := os.Create(dst)
	if err != nil {
		log.Fatalf("create input file: %v", err)
	}
	defer out.Close()
	if _, err := io.Copy(out, resp.Body); err != nil {
		log.Fatalf("copy payload: %v", err)
	}
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

func writeReceipt(rec receipt) {
	data, err := json.Marshal(rec)
	if err != nil {
		log.Fatalf("marshal receipt: %v", err)
	}
	if err := os.WriteFile("/work/receipt.json", data, 0o644); err != nil {
		log.Fatalf("write receipt: %v", err)
	}
}

func writeMetrics(metrics map[string]any) {
	data, err := json.Marshal(metrics)
	if err != nil {
		log.Fatalf("marshal metrics: %v", err)
	}
	if err := os.WriteFile("/work/metrics.json", data, 0o644); err != nil {
		log.Fatalf("write metrics: %v", err)
	}
}

func init() {
	log.SetFlags(0)
	log.SetPrefix("embed-runner: ")
}
