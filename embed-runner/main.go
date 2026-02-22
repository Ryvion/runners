package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

var allowedCmds = map[string]bool{
	"llama-embed":    true,
	"llama-cli":      true,
	"llama-quantize": true,
}

type JobSpec struct {
	InputFile  string   `json:"input_file"`
	OutputFile string   `json:"output_file"`
	Cmd        string   `json:"cmd"`
	Args       []string `json:"args"`
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

	if js.Cmd == "" {
		js.Cmd = "llama-embed"
	}
	cmdBase := filepath.Base(js.Cmd)
	if !allowedCmds[cmdBase] {
		log.Fatalf("blocked command: %q", js.Cmd)
	}

	if js.InputFile == "" {
		js.InputFile = "/work/input"
	}
	if js.OutputFile == "" {
		js.OutputFile = "/work/output"
	}

	args := make([]string, 0, len(js.Args))
	for _, a := range js.Args {
		if strings.ContainsAny(a, ";|&$`\\\"'<>()") && a != "{input}" && a != "{output}" {
			log.Fatalf("blocked argument with shell metacharacters: %q", a)
		}
		switch a {
		case "{input}":
			a = js.InputFile
		case "{output}":
			a = js.OutputFile
		}
		args = append(args, a)
	}

	if out, err := exec.Command(js.Cmd, args...).CombinedOutput(); err != nil {
		log.Fatalf("embed: %v\n%s", err, string(out))
	}

	outF, err := os.Open(js.OutputFile)
	if err != nil {
		log.Fatalf("open output: %v", err)
	}
	defer outF.Close()

	h := sha256.New()
	io.Copy(h, outF)

	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"output_hash": hex.EncodeToString(h.Sum(nil)),
		"ok":          true,
	})
}
