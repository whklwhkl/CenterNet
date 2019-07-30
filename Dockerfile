FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
#FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /workspace

COPY . /workspace

RUN pip install -r requirements.txt \
    && pip install --no-cache flask \
    && apt-get update \
    && apt-get install -y libglib2.0-0

EXPOSE 6666

CMD cd src/lib/models/networks/DCNv2/ \
    && ./make.sh \
    && cd /workspace \
    && python src/tools/worker.py