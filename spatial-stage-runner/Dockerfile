FROM golang:1.24-alpine AS build
ARG STAGE_KIND=spatial_stage
WORKDIR /src
COPY go.mod main.go ./
RUN go build -o /out/spatial-stage-runner ./main.go

FROM alpine:3.21
ARG STAGE_KIND=spatial_stage
ENV RYV_STAGE_KIND=${STAGE_KIND}
RUN adduser -D -H runner
COPY --from=build /out/spatial-stage-runner /usr/local/bin/spatial-stage-runner
USER runner
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/spatial-stage-runner"]
