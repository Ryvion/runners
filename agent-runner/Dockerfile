FROM python:3.11-slim

RUN useradd -r -s /usr/sbin/nologin runner

COPY run.py /app/run.py

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

STOPSIGNAL SIGTERM

USER runner
WORKDIR /work
ENTRYPOINT ["python", "-u", "/app/run.py"]
