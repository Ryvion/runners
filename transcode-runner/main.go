package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
)

type JobSpec struct {
	InputURL   string `json:"input_url"`
	OutputSink struct {
		Type string `json:"type"`
		URL  string `json:"url"`
	} `json:"output_sink"`
	Args []string `json:"args"`
}

func main() {
	specPath := "/work/job.json"
	f, err := os.Open(specPath)
	if err != nil {
		log.Fatalf("open job spec: %v", err)
	}
	var js JobSpec
	if err := json.NewDecoder(f).Decode(&js); err != nil {
		log.Fatalf("decode job spec: %v", err)
	}
	f.Close()

	// Validate input URL
	if js.InputURL != "" {
		u, err := url.Parse(js.InputURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
			log.Fatalf("invalid input_url: %q", js.InputURL)
		}
	}

	// Validate output sink URL
	if js.OutputSink.URL != "" {
		u, err := url.Parse(js.OutputSink.URL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
			log.Fatalf("invalid output_sink url: %q", js.OutputSink.URL)
		}
	}

	outPath := "/work/out.mp4"
	args := make([]string, 0, len(js.Args))
	for _, a := range js.Args {
		if strings.ContainsAny(a, ";|&$`\\\"'<>()") && a != "{input}" && a != "{output}" {
			log.Fatalf("blocked argument with shell metacharacters: %q", a)
		}
		switch a {
		case "{input}":
			a = js.InputURL
		case "{output}":
			a = outPath
		}
		args = append(args, a)
	}

	// Always use ffmpeg — never allow overriding the binary
	cmd := exec.Command("ffmpeg", args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		log.Fatalf("ffmpeg: %v\n%s", err, string(out))
	}

	outF, err := os.Open(outPath)
	if err != nil {
		log.Fatalf("open output: %v", err)
	}
	defer outF.Close()

	h := sha256.New()
	r := io.TeeReader(outF, h)
	req, err := http.NewRequest("PUT", js.OutputSink.URL, r)
	if err != nil {
		log.Fatalf("create PUT request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	_ = resp.Body.Close()

	sum := hex.EncodeToString(h.Sum(nil))
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"output_hash": sum,
		"ok":          true,
	})
}
