from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("/work/CKPTS/Qwen-VL-Chat-Int4", trust_remote_code=True)
config = AutoConfig.from_pretrained("/work/CKPTS/Qwen-VL-Chat-Int4", trust_remote_code=True)
config.quantization_config["use_exllama"] = False
# use cuda device
model = AutoModelForCausalLM.from_pretrained("/work/CKPTS/Qwen-VL-Chat-Int4", config=config, device_map="cpu", trust_remote_code=True).eval()

model.cuda()

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'http://localhost:8123/.images/4c33743cae5b14530c9bce576a1144e3.png'},
    {'text': 'Descripe the image in English:'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)