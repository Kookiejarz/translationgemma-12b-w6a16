from fastapi import FastAPI
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch, os, uvicorn

app = FastAPI()
MODEL_ID = "/model"
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer, model = None, None

def load_model():
    global tokenizer, model
    if model is not None:
        return tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoAWQForCausalLM.from_quantized(MODEL_ID, fuse_layers=True, trust_remote_code=True)
    return tokenizer, model

# 健康检查，Load Balancer 必须有
@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.post("/translate")
async def translate(request: dict):
    try:
        text = request.get("prompt", "").strip()
        if not text:
            return {"error": "请输入需要翻译的文本"}

        tok, mdl = load_model()
        messages = [{"role": "user", "content": f"Translate the following text to Chinese:\n{text}"}]
        input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(input_text, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = mdl.generate(**inputs, max_new_tokens=512, do_sample=False, repetition_penalty=1.1)

        translation = tok.decode(output[0][input_length:], skip_special_tokens=True).strip()
        return {"translation": translation}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    load_model()  # 启动时预加载
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "80")))