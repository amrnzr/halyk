import os
import json
import re
import gc
import torch
import pdfplumber
import pandas as pd
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from json_repair import repair_json

# 1. Структуры данных
class Covenant(BaseModel):
    clause: str = Field(description="Номер пункта договора, например '6.1'")
    type: str = Field(description="Тип правила: 'max_expense', 'min_revenue', или 'max_single_tx'")
    limit: float = Field(description="Числовое значение лимита из договора")

class ContractAnalysis(BaseModel):
    account_id: str = Field(description="Идентификатор аккаунта/заемщика, например 'ACC-7801'")
    covenants: list[Covenant]

# 2. Инициализация локальной модели
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Загрузка локальной модели LLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
print(f"Модель успешно загружена на {device.upper()}!")


def read_file_text(file_path: str) -> str:
    """Универсальное чтение текста из PDF, TXT и Excel"""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:10]:  # Ограничение первых 10 страниц
                text += (page.extract_text() or "") + "\n"

    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="cp1251", errors="ignore") as f:
                text = f.read()

    elif ext in [".xlsx", ".xls"]:
        try:
            excel_file = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in excel_file.items():
                text += f"\n--- Лист: {sheet_name} ---\n"
                text += df.to_string(index=False) + "\n"
        except Exception as e:
            print(f"Ошибка чтения Excel файла {os.path.basename(file_path)}: {e}")

    return text.strip()


def extract_json_from_text(text: str) -> dict:
    """Чистит и автоматически чинит поврежденный JSON от LLM"""
    # 1. Извлекаем кусок текста с JSON
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        match_raw = re.search(r'(\{.*?\})', text, re.DOTALL)
        json_str = match_raw.group(1) if match_raw else text

    # 2. Восстанавливаем синтаксис (запятые, кавычки, скобки)
    repaired_json_str = repair_json(json_str)

    # 3. Преобразуем в обычный Python dict
    return json.loads(repaired_json_str)


def extract_from_document(file_path: str) -> ContractAnalysis:
    """Главная функция обработки документа любого поддерживаемого формата"""
    text = read_file_text(file_path)

    if not text:
        print(f"⚠️ Не удалось извлечь текст из файла: {os.path.basename(file_path)}")
        return ContractAnalysis(account_id="UNKNOWN", covenants=[])

    # Ограничение длины символов для защиты от переполнения памяти
    if len(text) > 6000:
        text = text[:6000]

    prompt = f"""Ты — финансовый аналитик. Извлеки из текста договора ID заемщика/аккаунта (например: P1, P2, B1, ACC-P1 и т.д.) и условия ковенантов.

Текст договора:
{text}

Ответь СТРОГО в формате JSON без пояснений:
{{
  "account_id": "P1",
  "covenants": [
    {{
      "clause": "6.1",
      "type": "max_expense", 
      "limit": 300000.0
    }}
  ]
}}
Допустимые значения type: 'max_expense', 'min_revenue', 'max_single_tx'.
"""

    messages = [
        {"role": "system", "content": "Ты финансовый AI-ассистент, отдающий ответы строго в JSON."},
        {"role": "user", "content": prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Жесткая обрезка входной последовательности
    model_inputs = tokenizer(
        [prompt_text], 
        return_tensors="pt", 
        max_length=3072, 
        truncation=True
    ).to(model.device)

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        parsed_json = extract_json_from_text(response_text)
        result = ContractAnalysis(**parsed_json)

    except Exception as e:
        print(f"Ошибка обработки файла {os.path.basename(file_path)}: {e}")
        result = ContractAnalysis(account_id="UNKNOWN", covenants=[])

    finally:
        # Очистка оперативной памяти после каждого вызова
        del model_inputs
        if 'generated_ids' in locals():
            del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return result