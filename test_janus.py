import torch
from transformers import AutoModelForCausalLM
import sys
import warnings
warnings.filterwarnings("ignore")

model_path = "/project/jevans/tzhang3/models/Janus-Pro-7B"

vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, device_map="cpu", torch_dtype=torch.bfloat16
)
l = list(dict(vl_gpt.named_modules()).keys())
print("\n".join(l[:20]))
