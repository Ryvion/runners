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
	"strings"
	"time"
)

type jobSpec struct {
	Kind                 string   `json:"kind"`
	PayloadURL           string   `json:"payload_url"`
	InputURL             string   `json:"input_url"`
	InputFile            string   `json:"input_file"`
	OutputFile           string   `json:"output_file"`
	PipelineID           string   `json:"pipeline_id"`
	PipelineStages       []string `json:"pipeline_stages"`
	PipelineStageIndex   int      `json:"pipeline_stage_index"`
	Region               string   `json:"region"`
	RequiredCapabilities []string `json:"required_capabilities"`
	EstimatedDataGB      uint32   `json:"estimated_data_gb"`
	EstimatedScenes      uint32   `json:"estimated_scenes"`
}

type manifest struct {
	StageKind            string   `json:"stage_kind"`
	JobKind              string   `json:"job_kind,omitempty"`
	PipelineID           string   `json:"pipeline_id,omitempty"`
	PipelineStageIndex   int      `json:"pipeline_stage_index"`
	NextStage            string   `json:"next_stage,omitempty"`
	Region               string   `json:"region,omitempty"`
	RequiredCapabilities []string `json:"required_capabilities,omitempty"`
	SourceSHA256         string   `json:"source_sha256"`
	SourceBytes          int64    `json:"source_bytes"`
	EstimatedDataGB      uint32   `json:"estimated_data_gb,omitempty"`
	EstimatedScenes      uint32   `json:"estimated_scenes,omitempty"`
	ProcessedAt          string   `json:"processed_at"`
	Summary              string   `json:"summary"`
}

type receipt struct {
	OutputHash  string `json:"output_hash"`
	OK          bool   `json:"ok"`
	StageKind   string `json:"stage_kind"`
	SourceHash  string `json:"source_sha256,omitempty"`
	ArtifactURL string `json:"artifact_url,omitempty"`
}

func main() {
	stageKind := strings.TrimSpace(os.Getenv("RYV_STAGE_KIND"))
	if stageKind == "" {
		stageKind = "spatial_stage"
	}
	js := loadJob()
	inputPath, outputPath := resolvePaths(js)
	sourceHash, sourceBytes := prepareInput(js, inputPath)

	mf := manifest{
		StageKind:            stageKind,
		JobKind:              firstNonEmpty(js.Kind, stageKind),
		PipelineID:           strings.TrimSpace(js.PipelineID),
		PipelineStageIndex:   js.PipelineStageIndex,
		NextStage:            nextStage(js),
		Region:               strings.TrimSpace(js.Region),
		RequiredCapabilities: append([]string(nil), js.RequiredCapabilities...),
		SourceSHA256:         sourceHash,
		SourceBytes:          sourceBytes,
		EstimatedDataGB:      js.EstimatedDataGB,
		EstimatedScenes:      js.EstimatedScenes,
		ProcessedAt:          time.Now().UTC().Format(time.RFC3339),
		Summary:              fmt.Sprintf("%s accepted %d bytes and emitted a deterministic stage manifest", stageKind, sourceBytes),
	}

	writeJSON(outputPath, mf)
	outputHash, _ := hashFile(outputPath)
	writeJSON("/work/receipt.json", receipt{
		OutputHash: outputHash,
		OK:         true,
		StageKind:  stageKind,
		SourceHash: sourceHash,
	})
	writeJSON("/work/metrics.json", map[string]any{
		"engine":                stageKind,
		"source_bytes":          sourceBytes,
		"pipeline_stage_index":  js.PipelineStageIndex,
		"required_capabilities": js.RequiredCapabilities,
	})
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"output_hash": outputHash,
		"ok":          true,
		"stage_kind":  stageKind,
	})
}

func loadJob() jobSpec {
	f, err := os.Open("/work/job.json")
	if err != nil {
		log.Fatalf("open job spec: %v", err)
	}
	defer f.Close()
	var js jobSpec
	if err := json.NewDecoder(f).Decode(&js); err != nil {
		log.Fatalf("decode job spec: %v", err)
	}
	return js
}

func resolvePaths(js jobSpec) (string, string) {
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

func prepareInput(js jobSpec, inputPath string) (string, int64) {
	src := firstNonEmpty(strings.TrimSpace(js.PayloadURL), strings.TrimSpace(js.InputURL))
	if src != "" {
		downloadToFile(src, inputPath)
	}
	if _, err := os.Stat(inputPath); err != nil {
		log.Fatalf("input payload missing: %v", err)
	}
	return hashFile(inputPath)
}

func nextStage(js jobSpec) string {
	idx := js.PipelineStageIndex + 1
	if idx < 0 || idx >= len(js.PipelineStages) {
		return ""
	}
	return strings.TrimSpace(js.PipelineStages[idx])
}

func downloadToFile(srcURL, dst string) {
	client := &http.Client{Timeout: 15 * time.Minute}
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
		log.Fatalf("open file: %v", err)
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		log.Fatalf("hash file: %v", err)
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

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func init() {
	log.SetFlags(0)
	log.SetPrefix("spatial-stage-runner: ")
}
