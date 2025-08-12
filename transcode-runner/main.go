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
)

type JobSpec struct {
    InputURL   string `json:"input_url"`
    OutputSink struct {
        Type string `json:"type"` // "http-put"
        URL  string `json:"url"`
    } `json:"output_sink"`
    Args []string `json:"args"` // ffmpeg args; {input},{output}
}

func main() {
    dec := json.NewDecoder(os.Stdin)
    var js JobSpec
    if err := dec.Decode(&js); err != nil {
        log.Fatal(err)
    }

    outPath := "/tmp/out.mp4"
    args := []string{}
    for _, a := range js.Args {
        if a == "{input}" {
            a = js.InputURL
        }
        if a == "{output}" {
            a = outPath
        }
        args = append(args, a)
    }
    
    cmd := exec.Command("ffmpeg", args...)
    if out, err := cmd.CombinedOutput(); err != nil {
        log.Fatalf("ffmpeg: %v\n%s", err, string(out))
    }

    f, _ := os.Open(outPath)
    defer f.Close()
    
    h := sha256.New()
    r := io.TeeReader(f, h)
    req, _ := http.NewRequest("PUT", js.OutputSink.URL, r)
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