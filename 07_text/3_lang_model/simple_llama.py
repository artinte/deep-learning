import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

# model_name = 'NousResearch/Llama-2-7b-chat-hf'
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # bfloat16 is not supported on Mac CPU
    device_map={"": "cpu"}      # Run on CPU explicitly
)

prompt = "<s>[INST] Explain what a black hole is in one sentence. [/INST]"

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.1,)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
