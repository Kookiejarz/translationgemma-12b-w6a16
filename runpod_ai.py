import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 对应你在 RunPod Serverless 挂载卷时填写的路径
MODEL_DIR = "/runpod-volume/models/translategemma-w6a16"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None:
        print("Loading model from Network Volume...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
    return tokenizer, model

def handler(job):
    job_input = job['input']
    prompt = job_input.get("prompt", "Hello")
    
    tokenizer, model = load_model()
    
    inputs = tokenizer(f"Translate to Chinese: {prompt}", return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translation": response}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
