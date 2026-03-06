FROM golang:1.24-bookworm AS builder

WORKDIR /src
ARG TARGETARCH
COPY go.mod .
COPY main.go .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH:-amd64} go build -trimpath -ldflags='-s -w' -o /out/transcode-runner .

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/transcode-runner /usr/local/bin/transcode-runner

RUN useradd -r -s /usr/sbin/nologin runner

USER runner
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/transcode-runner"]
