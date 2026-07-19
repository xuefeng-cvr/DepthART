ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE_IMAGE}

ENV PIP_CONSTRAINT=""
COPY deploy/shared/requirements.txt /tmp/depthart-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/depthart-requirements.txt

WORKDIR /workspace/DepthART
