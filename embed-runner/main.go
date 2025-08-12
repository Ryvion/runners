package main

import (
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "io"
    "log"
    "os"
    "os/exec"
)

type JobSpec struct {
    InputFile  string   `json:"input_file"`
    OutputFile string   `json:"output_file"`
    Cmd        string   `json:"cmd"`  // "llama-embed"
    Args       []string `json:"args"` // include {input},{output}
}

func main() {
    var js JobSpec
    if err := json.NewDecoder(os.Stdin).Decode(&js); err != nil {
        log.Fatal(err)
    }
    
    if js.Cmd == "" {
        js.Cmd = "llama-embed"
    }
    
    args := []string{}
    for _, a := range js.Args {
        if a == "{input}" {
            a = js.InputFile
        }
        if a == "{output}" {
            a = js.OutputFile
        }
        args = append(args, a)
    }
    
    if out, err := exec.Command(js.Cmd, args...).CombinedOutput(); err != nil {
        log.Fatalf("embed: %v\n%s", err, string(out))
    }
    
    f, _ := os.Open(js.OutputFile)
    defer f.Close()
    
    h := sha256.New()
    io.Copy(h, f)
    
    _ = json.NewEncoder(os.Stdout).Encode(map[string]any{
        "output_hash": hex.EncodeToString(h.Sum(nil)),
        "ok":          true,
    })
}