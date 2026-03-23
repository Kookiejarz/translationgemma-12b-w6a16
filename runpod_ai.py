import runpod
import torch
import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 你的仓库 ID
MODEL_ID = "Kennethhhh/translationgemma-12b-w6a16"
HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None


def load_model():
    global tokenizer, model
    if model is not None:
        return tokenizer, model

    print(f"🚀 开始从 Hugging Face 加载 AWQ 模型: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )

    model = AutoAWQForCausalLM.from_quantized(
        MODEL_ID,
        fuse_layers=True,       # 融合算子，推理更快
        trust_remote_code=True,
        token=HF_TOKEN
    )

    print("✅ 模型已成功载入显存")
    return tokenizer, model


def handler(job):
    try:
        job_input = job.get("input", {})
        text = job_input.get("prompt", "").strip()

        if not text:
            return {"error": "请输入需要翻译的文本"}

        tokenizer, model = load_model()

        # 用 chat template 触发指令跟随能力
        # 如果你微调时用的是自定义格式，把这里换成对应格式
        messages = [
            {
                "role": "user",
                "content": f"Translate the following text to Chinese:\n{text}"
            }
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,        # 翻译任务关闭随机性
                repetition_penalty=1.1  # 防止输出死循环
            )

        # 只截取新生成的 token，避免字符串 replace 的不稳定问题
        new_tokens = output[0][input_length:]
        translation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {"translation": translation}

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()  # 方便在 RunPod 日志里定位问题
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})