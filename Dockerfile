FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt \
    && apt-get update \
    && apt-get install -y libglib2.0-0

ENV root /workspace

ENV coco=${root}/data/coco/PythonAPI \
    dconv=${root}/src/lib/models/networks/DCNv2 \
    nms=${root}/src/lib/external/