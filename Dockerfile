FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

WORKDIR /
RUN pip install runpod transformers accelerate safetensors sentencepiece

# 注意：这里的文件名要和你实际写的 Python 脚本名一致
COPY runpod_ai.py / 

CMD ["python", "-u", "/runpod_ai.py"]
