import os
import json
import re
import pandas as pd
from extract_covenants13 import extract_from_document, read_file_text
from evaluate13 import process_ledger, load_ledger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_account_scenario_map(df: pd.DataFrame) -> dict:
    """
    Строит надёжное соответствие account_id (как он записан в леджере,
    например 'ACC-7801') -> scenario_id ('P1'), используя префикс txn_id.
    Это заменяет прежнее угадывание scenario_id по regex-поиску в свободном
    тексте документа, которое могло ошибочно сработать на постороннем
    упоминании "P1" где-то в тексте.
    """
    cols = {c.lower(): c for c in df.columns}
    txn_col = cols.get("txn_id") or cols.get("transaction_id")
    acc_col = cols.get("account_id") or cols.get("account")

    mapping = {}
    conflicts = {}
    for _, row in df[[txn_col, acc_col]].drop_duplicates().iterrows():
        txn = str(row[txn_col])
        acc = str(row[acc_col]).strip().upper()
        m = re.match(r"^TXN-([A-Za-z0-9]+)-", txn)
        if not m:
            continue
        scenario = m.group(1).upper()
        if acc in mapping and mapping[acc] != scenario:
            conflicts.setdefault(acc, set()).update({mapping[acc], scenario})
        else:
            mapping[acc] = scenario

    if conflicts:
        print(f"⚠️ account_id с противоречивыми scenario_id (проверьте вручную): {conflicts}")

    return mapping


def find_scenario_id(account_id: str, file_text: str, valid_scenarios: list,
                      account_scenario_map: dict) -> str:
    """
    Определяет scenario_id для документа.
    Порядок приоритета:
      1. Точное соответствие account_id -> scenario_id из леджера (надёжно).
      2. account_id уже сам является scenario_id (например 'P1').
      3. Регекс-поиск scenario_id внутри строки account_id.
      4. (Крайний случай) регекс-поиск в первых 2000 символах текста документа.
    """
    if account_id:
        acc_norm = str(account_id).strip().upper()
        if acc_norm in account_scenario_map:
            sc = account_scenario_map[acc_norm]
            if sc in valid_scenarios:
                return sc
            print(f"⚠️ scenario '{sc}' из леджера отсутствует в шаблоне для account_id {acc_norm}")

    clean_acc = re.sub(r'^(ACC|ACCOUNT)[-_]?', '', str(account_id or ''), flags=re.IGNORECASE).strip().upper()
    if clean_acc in valid_scenarios:
        return clean_acc

    for sc in valid_scenarios:
        if re.search(rf'\b{re.escape(sc)}\b', str(account_id or ''), re.IGNORECASE):
            return sc

    for sc in valid_scenarios:
        if re.search(rf'\b{re.escape(sc)}\b', file_text[:2000], re.IGNORECASE):
            print(f"ℹ️ scenario_id '{sc}' определён по тексту документа (менее надёжно, проверьте вручную)")
            return sc

    return None


def normalize_clause(clause: str) -> str:
    """'Пункт 6.1.' / '6.1 ' / '6.1' -> '6.1' — чтобы несовпадение форматирования
    не роняло валидный ковенант."""
    return re.sub(r'[^0-9.]', '', str(clause or '')).strip('.')


def main():
    template_path = os.path.join(BASE_DIR, "submission_template.json")
    ledger_path = os.path.join(BASE_DIR, "master_ledger_2025.csv")
    docs_dir = os.path.join(BASE_DIR, "documents")
    cache_path = os.path.join(BASE_DIR, "extracted_cache.json")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Файл шаблона не найден по пути: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    valid_scenarios = list(template["answers"].keys())

    template["team"] = "onemanarmy"
    template["contact_email"] = "amir.nazarov07@gmail.com"
    template["model"] = "Qwen2.5-1.5B-Instruct"

    # ФИКС: леджер грузился заново на КАЖДЫЙ ковенант внутри process_ledger.
    # Теперь читаем один раз и передаём DataFrame.
    ledger_df = load_ledger(ledger_path)
    account_scenario_map = build_account_scenario_map(ledger_df)
    print(f"Построено соответствий account_id -> scenario_id: {len(account_scenario_map)}")

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # Добавлен .docx
    SUPPORTED_EXTENSIONS = ('.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls')

    unmatched_files = []

    for filename in os.listdir(docs_dir):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = os.path.join(docs_dir, filename)
            print(f"\n--- Обработка файла: {filename} ---")

            if filename in cache:
                print("Данные взяты из кэша.")
                extracted_data = cache[filename]
            else:
                extracted = extract_from_document(file_path)
                extracted_data = extracted.model_dump()
                cache[filename] = extracted_data

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)

            account_id = extracted_data.get("account_id")
            covenants = extracted_data.get("covenants", [])

            if not covenants:
                # либо документ не про ковенанты (KYC/аудит и т.п.), либо LLM не справилась
                continue

            file_text = read_file_text(file_path)
            scenario_id = find_scenario_id(account_id, file_text, valid_scenarios, account_scenario_map)
            print(f"Извлеченный account_id: '{account_id}' -> Определен Scenario ID: '{scenario_id}'")

            if not scenario_id:
                print(f"⚠️ Не удалось привязать файл {filename} к сценариям из шаблона!")
                unmatched_files.append(filename)
                continue

            for cov in covenants:
                clause = cov.get("clause")
                clause_norm = normalize_clause(clause)

                target_keys = template["answers"].get(scenario_id, {})
                matched_key = clause if clause in target_keys else next(
                    (k for k in target_keys if normalize_clause(k) == clause_norm), None
                )

                if matched_key is None:
                    print(f"⚠️ Пункт '{clause}' не найден в шаблоне для {scenario_id} — пропущен")
                    continue

                result = process_ledger(ledger_df, scenario_id, cov)
                template["answers"][scenario_id][matched_key] = result
                print(f"  [Сценарий {scenario_id} | Пункт {matched_key}] -> "
                      f"Status: {result['status']}, Actual: {result['actual']}, "
                      f"Evidence: {result['evidence_txn_id']}")

    if unmatched_files:
        print(f"\n⚠️ Документы без сценария (проверьте вручную): {unmatched_files}")

    # Отчёт по незаполненным ячейкам — легко пропустить пустую ячейку, а она стоит 0
    missing = []
    for sc, clauses in template["answers"].items():
        for clause, cell in clauses.items():
            if cell.get("status") not in ("COMPLIANT", "BREACH"):
                missing.append(f"{sc}/{clause}")
    if missing:
        print(f"\n⚠️ Незаполненные/некорректные ячейки ({len(missing)}): {missing}")

    output_path = os.path.join(BASE_DIR, "submission.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"\nГотово! Файл сохранён в: {output_path}")


if __name__ == "__main__":
    main()
