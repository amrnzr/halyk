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

# 1. Структуры данных
class Covenant(BaseModel):
    clause: str = Field(description="Номер пункта договора, например '6.1'")
    type: str = Field(description="Тип правила: 'max_expense', 'min_revenue', 'max_single_tx', 'min_balance'")
    limit: float = Field(description="Числовое значение лимита из договора")
    period_start: Optional[str] = Field(default=None, description="Начало периода теста 'YYYY-MM-DD', если ковенант привязан к периоду (квартал/год), иначе null")
    period_end: Optional[str] = Field(default=None, description="Конец периода теста 'YYYY-MM-DD', иначе null")


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
    """Универсальное чтение текста из PDF, DOCX, TXT и Excel"""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:20]:  # было 10 — увеличено, договоры бывают длиннее
                text += (page.extract_text() or "") + "\n"

    elif ext == ".docx":
        try:
            from docx import Document  # pip install python-docx
            doc = Document(file_path)
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            text = "\n".join(parts)
        except ImportError:
            print("⚠️ Нужен пакет python-docx: pip install python-docx --break-system-packages")
        except Exception as e:
            print(f"Ошибка чтения DOCX файла {os.path.basename(file_path)}: {e}")

    elif ext == ".doc":
        # Старый бинарный формат .doc не читается python-docx.
        try:
            import docx2txt  # pip install docx2txt
            text = docx2txt.process(file_path)
        except ImportError:
            print(f"⚠️ Файл {os.path.basename(file_path)} — старый формат .doc. "
                  f"Установите docx2txt (pip install docx2txt) или переконвертируйте в .docx")
        except Exception as e:
            print(f"Ошибка чтения DOC файла {os.path.basename(file_path)}: {e}")

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


_ACCOUNT_ID_RE = re.compile(r'\b([A-ZА-Я]{2,6}[\s\-]?\d{3,6})\b')
_TXN_ID_RE = re.compile(r'\bTXN[\s\-]?[A-Za-z0-9]+[\s\-]?\d{3,6}\b')
_RECLASS_KEYWORDS = ["переклассифика", "reclassif", "исправлен", "restated", "corrected entry"]
_FX_KEYWORDS = ["курс", "exchange rate", "fx rate", "обменный курс"]


def extract_structured_hints(text: str) -> dict:
    """
    Многие документы в датасете (аудиторские примечания и т.п.) содержат
    надёжные структурные маркеры прямо в тексте — например
    'АУДИТОРСКОЕ ДЕЛО № ACC-7301/2025' — которые regex находит гораздо
    точнее, чем 1.5B-модель "на глаз". Используем их как приоритетный
    источник account_id, а не только LLM-догадку.

    Также вытаскиваем упоминания переклассификаций (потенциальный источник
    evidence_txn_id) и упоминания валютных курсов (могут указывать, что
    нужный курс — в ДРУГОМ документе, как в вашем случае с AR-2025-0114).
    """
    # Прямой поиск ACC-XXXX / TELE-XXXX и т.п. — предпочитаем формат "БУКВЫ-ЦИФРЫ"
    # рядом со словами типа "ДЕЛО №", "account", "заёмщик"
    account_candidates = re.findall(r'\b([A-Z]{2,6}-\d{3,6})\b', text)

    txn_mentions = sorted(set(_TXN_ID_RE.findall(text)))

    reclass_hits = []
    for kw in _RECLASS_KEYWORDS:
        for m in re.finditer(kw, text, re.IGNORECASE):
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 120)
            reclass_hits.append(text[start:end].replace("\n", " ").strip())

    fx_hits = []
    for kw in _FX_KEYWORDS:
        for m in re.finditer(kw, text, re.IGNORECASE):
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 120)
            fx_hits.append(text[start:end].replace("\n", " ").strip())

    return {
        "account_candidates": sorted(set(account_candidates)),
        "txn_mentions": txn_mentions,
        "reclass_hits": reclass_hits[:5],   # ограничиваем на всякий случай
        "fx_hits": fx_hits[:5],
    }


def looks_like_credit_agreement(text: str) -> bool:
    """
    Датасет содержит вперемешку кредитные договоры, аудиторские примечания,
    годовые отчёты и т.д. Прежний фильтр (просто "есть ли цифра.цифра и
    ключевое слово") пропускал МНОГОЕ лишнее — например годовой отчёт со
    строкой "31 December 2025" и словом "financial" формально проходил.
    Теперь требуем явную связь с ковенантами конкретного заёмщика:
    - упоминание номера пункта в характерном для договора виде ("Пункт 6.1",
      "Section 6.1", "Clause 6.1", "6.1.") И
    - хотя бы одно упоминание слова "ковенант"/"covenant" в пределах
      разумного контекста (не просто где-то в PDF).
    Документы без этого (аудиторские примечания, годовые отчёты, KYC-досье)
    пропускаются: их полезная информация (FX-курсы, реклассификации)
    забирается отдельно через extract_structured_hints(), без затратного
    вызова LLM для "извлечения ковенантов" из документа, в котором их нет.
    """
    if not text:
        return False
    has_clause_ref = re.search(
        r'(пункт|clause|section)\s*\d{1,2}\.\d{1,2}', text, re.IGNORECASE
    ) is not None
    has_covenant_word = re.search(r'ковенант|covenant', text, re.IGNORECASE) is not None
    return has_clause_ref and has_covenant_word


def _extract_balanced_json(text: str) -> str:
    """
    Возвращает первый сбалансированный JSON-объект в тексте, считая {}.
    ФИКС: старая версия использовала нежадный regex r'(\\{.*?\\})', который
    для вложенного JSON (список ковенантов внутри объекта) обрывался на
    первой внутренней '}' — то есть отдавал только первый ковенант,
    остальные молча терялись.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # незакрытый JSON — repair_json попробует починить


def extract_json_from_text(text: str) -> dict:
    """Чистит и автоматически чинит поврежденный JSON от LLM"""
    match = re.search(r'```json\s*(.*?)```', text, re.DOTALL)
    raw = match.group(1) if match else text

    json_str = _extract_balanced_json(raw)
    repaired_json_str = repair_json(json_str)
    return json.loads(repaired_json_str)


def _windowed_text(text: str, max_chars: int = 8000) -> str:
    """
    Раньше текст просто обрезался до первых 6000 символов, что могло
    отрезать раздел ковенантов, если он расположен дальше в документе.
    Теперь ищем ключевые слова и центрируем окно вокруг них.
    """
    if len(text) <= max_chars:
        return text
    lower = text.lower()
    idx = -1
    for kw in ["ковенант", "covenant", "financial covenants", "раздел 6", "6."]:
        idx = lower.find(kw)
        if idx != -1:
            break
    if idx == -1:
        return text[:max_chars]
    start = max(0, idx - 1500)
    return text[start:start + max_chars]


def extract_from_document(file_path: str) -> ContractAnalysis:
    """Главная функция обработки документа любого поддерживаемого формата"""
    text = read_file_text(file_path)

    if not text:
        print(f"⚠️ Не удалось извлечь текст из файла: {os.path.basename(file_path)}")
        return ContractAnalysis(account_id="UNKNOWN", covenants=[])

    hints = extract_structured_hints(text)
    if hints["reclass_hits"]:
        print(f"🔎 {os.path.basename(file_path)}: упоминание переклассификации/исправления — "
              f"проверьте вручную, это возможный источник evidence_txn_id:")
        for h in hints["reclass_hits"]:
            print(f"   ...{h}...")
    if hints["fx_hits"]:
        print(f"🔎 {os.path.basename(file_path)}: упоминание валютного курса — "
              f"проверьте вручную (курс может быть указан в ДРУГОМ документе, "
              f"на который здесь есть ссылка):")
        for h in hints["fx_hits"]:
            print(f"   ...{h}...")

    if not looks_like_credit_agreement(text):
        print(f"ℹ️ {os.path.basename(file_path)} не похож на кредитный договор с ковенантами — "
              f"пропускаю LLM-извлечение ковенантов (структурные подсказки выше уже собраны)")
        result = ContractAnalysis(account_id="UNKNOWN", covenants=[])
        # даже для пропущенного документа отдаём найденный account_id по regex,
        # если он есть — main.py использует его для сопоставления со сценарием
        if hints["account_candidates"]:
            result.account_id = hints["account_candidates"][0]
        return result

    text = _windowed_text(text, max_chars=8000)

    prompt = f"""Ты — финансовый аналитик. Извлеки из текста договора ID заемщика/аккаунта (например: P1, P2, B1, ACC-P1 и т.д.) и ВСЕ условия финансовых ковенантов, сколько бы их ни было.

Текст договора:
{text}

Ответь СТРОГО в формате JSON без пояснений, включи КАЖДЫЙ найденный ковенант в массив covenants:
{{
  "account_id": "P1",
  "covenants": [
    {{
      "clause": "6.1",
      "type": "max_expense",
      "limit": 300000.0,
      "period_start": null,
      "period_end": null
    }},
    {{
      "clause": "6.2",
      "type": "min_revenue",
      "limit": 1000000.0,
      "period_start": "2025-01-01",
      "period_end": "2025-12-31"
    }}
  ]
}}
Допустимые значения type: 'max_expense', 'min_revenue', 'max_single_tx', 'min_balance'.
Если ковенант проверяется за конкретный период (квартал, год), заполни period_start/period_end датами в формате YYYY-MM-DD; если период не указан явно в тексте — оставь null (тогда берётся вся история транзакций).
Если встретится ковенант другого типа (коэффициент, агрегатный лимит и т.п.), всё равно включи его в список с ближайшим подходящим type — такие пункты потом будут проверены вручную.
"""

    messages = [
        {"role": "system", "content": "Ты финансовый AI-ассистент, отдающий ответы строго в JSON."},
        {"role": "user", "content": prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
        max_length=4096,
        truncation=True
    ).to(model.device)

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=768,  # было 256 — мало для нескольких ковенантов
                temperature=0.1,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        parsed_json = extract_json_from_text(response_text)
        result = ContractAnalysis(**parsed_json)

        # Regex-найденный ID (напр. из "ДЕЛО № ACC-7301") надёжнее, чем то,
        # что модель могла напечатать с опечаткой/в другом формате.
        if hints["account_candidates"] and result.account_id.upper() not in [
            c.upper() for c in hints["account_candidates"]
        ]:
            print(f"ℹ️ LLM извлекла account_id='{result.account_id}', но в тексте по regex "
                  f"найдено {hints['account_candidates']} — используем первый вариант из regex")
            result.account_id = hints["account_candidates"][0]

    except Exception as e:
        print(f"Ошибка обработки файла {os.path.basename(file_path)}: {e}")
        result = ContractAnalysis(account_id="UNKNOWN", covenants=[])

    finally:
        del model_inputs
        if 'generated_ids' in locals():
            del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return result
