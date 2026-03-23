from fastapi import FastAPI
from fastapi.responses import Response
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch, os, uvicorn, time

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
    model = AutoAWQForCausalLM.from_quantized(
        MODEL_ID,
        fuse_layers=True,
        trust_remote_code=True
    )
    return tokenizer, model


def detect_language(text: str) -> str:
    """简单判断是否含中文字符，决定翻译方向"""
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return "zh"
    return "en"


def build_messages(prompt: str | None, messages: list | None) -> list:
    """
    支持两种输入方式：
    1. {"prompt": "..."} - 自动检测语言，构建翻译指令
    2. {"messages": [{"role": "user", "content": "..."}]} - 直接传入，完全自定义
    """
    if messages:
        return messages

    lang = detect_language(prompt)
    if lang == "zh":
        instruction = f"Translate the following text to English:\n{prompt}"
    else:
        instruction = f"Translate the following text to Chinese:\n{prompt}"

    return [{"role": "user", "content": instruction}]


# 健康检查：模型未就绪返回 204，就绪后返回 200
@app.get("/ping")
async def ping():
    if model is None:
        return Response(status_code=204)
    return {"status": "healthy"}


@app.post("/translate")
async def translate(request: dict):
    try:
        prompt = request.get("prompt", "").strip()
        messages = request.get("messages", None)

        if not prompt and not messages:
            return {"error": "请提供 prompt 或 messages"}

        tok, mdl = load_model()

        chat_messages = build_messages(prompt or None, messages)
        input_text = tok.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(input_text, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            output = mdl.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1
            )
        translation = tok.decode(output[0][input_length:], skip_special_tokens=True).strip()
        elapsed = time.time() - start

        return {
            "translation": translation,
            "inference_time_seconds": round(elapsed, 2)
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    load_model()  # 启动时预加载，加载完成前 /ping 返回 204
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "80")))