FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

WORKDIR /workspace

COPY . .

RUN pip install -r requirements.txt \
&& bash setup.sh