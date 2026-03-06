FROM python:3.11-slim
ARG STAGE_KIND=spatial_stage
ENV RYV_STAGE_KIND=${STAGE_KIND}
ENV MPLBACKEND=Agg
WORKDIR /work
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.4 scipy==1.15.2 opencv-python-headless==4.11.0.86 trimesh==4.6.5 matplotlib==3.10.1 pillow==11.1.0 networkx==3.4.2
COPY run.py /usr/local/bin/spatial-stage-runner
ENTRYPOINT ["python", "/usr/local/bin/spatial-stage-runner"]
