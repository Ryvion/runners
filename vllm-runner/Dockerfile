ARG VLLM_BASE_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_BASE_IMAGE}
COPY run.py /app/run.py
WORKDIR /work
ENTRYPOINT ["python", "-u", "/app/run.py"]
