import torch
import pathlib
import numpy as np
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from hf_olmo.modeling_olmo import OLMoForCausalLM
from hf_olmo.configuration_olmo import OLMoConfig

# 避免 torch.load 出错
torch.serialization.add_safe_globals([
    pathlib.PosixPath,
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.UInt32DType,
])

def load_model_and_ckpt_unsharded():
    # 1. 加载 Qwen tokenizer
    tokenizer_path = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 2. 读取 YAML 配置
    yaml_path = "/mnt/zzb/workspace/three_data/train_codes/gj/OLMo/configs/official-0425/OLMo2-1B-stage1.yaml"
    cfg = OmegaConf.load(yaml_path)

    # 3. 计算 mlp_hidden_size 并构造 config
    cfg.model["mlp_hidden_size"] = cfg.model.d_model * cfg.model.mlp_ratio
    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    config = OLMoConfig.from_dict(model_config_dict)

    # 4. 初始化模型结构
    model = OLMoForCausalLM(config)

    # 5. 直接加载 unsharded checkpoint 文件（修改路径为你的实际路径）
    ckpt_path = "/mnt/zzb/workspace/three_data/train_codes/gj/OLMo/tmp/Checkpoints/Qwen/OLMo2-1B-stage1/step7400-unsharded/model.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # 6. 如果缺失 model. 前缀，加上
    if not any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {"model." + k: v for k, v in state_dict.items()}

    # 7. 加载权重到模型
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to("cuda")

    return model, tokenizer

def generate_reply(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=2
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 去除重复 prompt
    if full_text.startswith(prompt):
        full_text = full_text[len(prompt):].strip()

    # 如果第一个字符是标点，去掉它
    if full_text and full_text[0] in ['，', '。', '！', '？', ',', '.', '!', '?']:
        full_text = full_text[1:].strip()

    # 截断到第一个终止标点
    END_TOKENS = ['。', '！', '？', '.', '!', '?']
    for token in END_TOKENS:
        if token in full_text:
            idx = full_text.index(token)
            return full_text[:idx+1].strip()

    return full_text[:max_new_tokens].strip()





def interactive_loop(model, tokenizer):
    print("🔁 进入 OLMo 对话模式，输入 q 退出。")
    while True:
        user_input = input("\n🧑‍💻 你：")
        if user_input.lower().strip() in {"q", "quit", "exit"}:
            print("👋 再见！")
            break
        reply = generate_reply(model, tokenizer, user_input)
        print(f"\n🤖 OLMo：{reply}")

if __name__ == "__main__":
    print("🚀 正在加载模型和 tokenizer，请稍候...")
    model, tokenizer = load_model_and_ckpt_unsharded()
    interactive_loop(model, tokenizer)
