import os
import json
import re
import gc
import torch
import pdfplumber
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from json_repair import repair_json

class Covenant(BaseModel):
    clause: str = Field(description="Номер пункта договора, например '6.1'")
    type: str = Field(description="Тип правила: 'max_expense', 'min_revenue', 'max_single_tx', 'min_balance', 'external_metric'")
    limit: float = Field(description="Числовое значение лимита из договора")
    period_start: Optional[str] = Field(default=None, description="Начало периода")
    period_end: Optional[str] = Field(default=None, description="Конец периода")

class ContractAnalysis(BaseModel):
    account_id: str = Field(description="Идентификатор аккаунта/заемщика")
    covenants: list[Covenant]

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
print("Загрузка локальной модели LLM...")
tokenizer = AutoTokenizer.frompretrained(MODEL_NAME) if "frompretrained" in dir(AutoTokenizer) else AutoTokenizer.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

def read_file_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:30]:  # Увеличили охват страниц
                text += (page.extract_text() or "") + "\n"
    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f: text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="cp1251", errors="ignore") as f: text = f.read()
    # (пропущены docx/xlsx для компактности вывода, можете оставить свою версию read_file_text целиком)
    return text.strip()

_RECLASS_KEYWORDS = ["переклассифика", "reclassif", "исправлен", "restated"]
_FX_RATE_NUMBER_RE = re.compile(r'(курс|exchange rate)[^.\n]{0,60}(\d[\d.,]*\s*(USD|EUR|KZT))', re.IGNORECASE)
_SUPERSEDED_KEYWORDS = ["недействующая редакция", "не применяется", "superseded"]

def extract_structured_hints(text: str) -> dict:
    account_candidates = re.findall(r'\b([A-Z]{2,6}-\d{3,6})\b', text)
    is_superseded = any(kw in text.lower() for kw in _SUPERSEDED_KEYWORDS)
    return {
        "account_candidates": sorted(set(account_candidates)),
        "is_superseded": is_superseded,
    }

def _extract_balanced_json(text: str) -> str:
    start = text.find("{")
    if start == -1: return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[start:i + 1]
    return text[start:]

def get_clause_window(text: str, clause: str, window_size: int = 2500) -> str:
    """Вырезает узкий фрагмент вокруг конкретного пункта для точечного извлечения"""
    pattern = rf"(?i)(?:пункт|clause|section|п\.)?\s*{re.escape(clause)}\b"
    match = re.search(pattern, text)
    if match:
        start = max(0, match.start() - 300)
        end = min(len(text), match.end() + window_size)
        return text[start:end]
    return text[:window_size]

def extract_from_document(file_path: str) -> ContractAnalysis:
    text = read_file_text(file_path)
    if not text:
        return ContractAnalysis(account_id="UNKNOWN", covenants=[])

    hints = extract_structured_hints(text)
    account_id = hints["account_candidates"][0] if hints["account_candidates"] else "UNKNOWN"
    result = ContractAnalysis(account_id=account_id, covenants=[])

    if hints["is_superseded"]:
        print(f"🚫 {os.path.basename(file_path)}: недействующая редакция.")
        return result

    # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Ищем пункты 5.x, 6.x, 7.x через regex ---
    # Собираем уникальные номера пунктов (например '6.1', '6.2', '6.3')
    clause_matches = re.findall(r'(?i)(?:пункт|clause|section|п\.)\s*([567]\.\d{1,2})\b', text)
    unique_clauses = list(dict.fromkeys(clause_matches)) # Сохраняем порядок
    
    if not unique_clauses:
        return result

    print(f"Найдено пунктов для обработки: {unique_clauses[:10]}")

    # Перебираем найденные пункты и вызываем LLM на КАЖДЫЙ по отдельности
    for clause in unique_clauses[:10]: # Ограничитель, чтобы не зависнуть
        window = get_clause_window(text, clause)
        
        prompt = f"""Ты финансовый аналитик. Тебе дан фрагмент кредитного договора.
Твоя задача — извлечь параметры ТОЛЬКО для ковенанта, описанного в пункте {clause}.

Фрагмент:
{window}

Правила type:
1. Расходы заемщика -> "max_expense"
2. Выручка -> "min_revenue"
3. Ограничение разовой транзакции -> "max_single_tx"
4. Ссылка на отчетность материнской компании (Группы) -> "external_metric"

Ответь СТРОГО JSON без пояснений:
{{
  "clause": "{clause}",
  "type": "тип",
  "limit": 1000000.0,
  "period_start": null,
  "period_end": null
}}"""
        
        messages = [{"role": "system", "content": "Отвечай строго в JSON."}, {"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([prompt_text], return_tensors="pt", max_length=2048, truncation=True).to(model.device)
        
        try:
            with torch.no_grad():
                generated_ids = model.generate(**model_inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            json_str = _extract_balanced_json(response_text)
            parsed = json.loads(repair_json(json_str))
            
            # Если лимит найден или тип определен, добавляем в список
            if parsed.get("limit", 0) > 0 or parsed.get("type") != "UNKNOWN":
                result.covenants.append(Covenant(**parsed))
                print(f"  -> Извлечен {clause}: {parsed.get('type')}")
        except Exception as e:
            pass # Игнорируем пункты, где LLM сломалась, идем к следующему
        finally:
            del model_inputs, generated_ids
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

    return result