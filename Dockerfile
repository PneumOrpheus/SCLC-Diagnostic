# Has to match the cuda version from your machine
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL maintainer="Rafael Oversand <rhoversa@stud.ntnu.no>"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/SCLC-Classification

COPY environment.yaml requirements.txt ./

RUN conda env update -n base --file environment.yaml && \
    conda clean -ya

RUN mkdir -p /workspace/SCLC-Classification/resources

CMD ["/bin/bash"]