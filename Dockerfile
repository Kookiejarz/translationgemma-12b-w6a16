FROM FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN pip install runpod transformers autoawq huggingface_hub fastapi uvicorn

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='Kennethhhh/translationgemma-12b-w6a16', token=os.environ['HF_TOKEN'], local_dir='/model')"

COPY runpod_ai.py .
CMD ["python", "runpod_ai.py"]