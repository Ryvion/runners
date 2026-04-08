ARG VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504
FROM ${VLLM_BASE_IMAGE}
COPY run.py /app/run.py
WORKDIR /work
ENTRYPOINT ["python", "-u", "/app/run.py"]
