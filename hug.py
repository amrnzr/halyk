import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загружаем модель и токенизатор (автоматически скачаются при первом запуске)
model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto" # Использует GPU, если есть, иначе CPU
)

def generate_answer(prompt):
    messages = [
        {"role": "system", "content": "Ты — финансовый аналитик. Извлеки данные в формате JSON."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response