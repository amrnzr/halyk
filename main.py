import os
import json
import re
import pandas as pd
from extract_covenants11 import extract_from_document, read_file_text
from evaluate11 import process_ledger, load_ledger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_account_scenario_map(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}
    txn_col = cols.get("txn_id") or cols.get("transaction_id")
    acc_col = cols.get("account_id") or cols.get("account")
    mapping = {}
    for _, row in df[[txn_col, acc_col]].drop_duplicates().iterrows():
        txn = str(row[txn_col])
        acc = str(row[acc_col]).strip().upper()
        m = re.match(r"^TXN-([A-Za-z0-9]+)-", txn)
        if m: mapping[acc] = m.group(1).upper()
    return mapping

def find_scenario_id(account_id: str, file_text: str, valid_scenarios: list, account_scenario_map: dict) -> str:
    if account_id:
        acc_norm = str(account_id).strip().upper()
        if acc_norm in account_scenario_map and account_scenario_map[acc_norm] in valid_scenarios:
            return account_scenario_map[acc_norm]
    clean_acc = re.sub(r'^(ACC|ACCOUNT)[-_]?', '', str(account_id or ''), flags=re.IGNORECASE).strip().upper()
    if clean_acc in valid_scenarios: return clean_acc
    for sc in valid_scenarios:
        if re.search(rf'\b{re.escape(sc)}\b', str(account_id or ''), re.IGNORECASE): return sc
    for sc in valid_scenarios:
        if re.search(rf'\b{re.escape(sc)}\b', file_text[:2000], re.IGNORECASE): return sc
    return None

def normalize_clause(clause: str) -> str:
    return re.sub(r'[^0-9.]', '', str(clause or '')).strip('.')

def main():
    template_path = os.path.join(BASE_DIR, "submission_template.json")
    ledger_path = os.path.join(BASE_DIR, "master_ledger_2025.csv")
    docs_dir = os.path.join(BASE_DIR, "documents")
    cache_path = os.path.join(BASE_DIR, "extracted_cache.json")

    with open(template_path, "r", encoding="utf-8") as f: template = json.load(f)
    valid_scenarios = list(template["answers"].keys())

    ledger_df = load_ledger(ledger_path)
    account_scenario_map = build_account_scenario_map(ledger_df)

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f: cache = json.load(f)

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
            if not covenants: continue

            file_text = read_file_text(file_path)
            scenario_id = find_scenario_id(account_id, file_text, valid_scenarios, account_scenario_map)
            
            if not scenario_id:
                unmatched_files.append(filename)
                continue

            for cov in covenants:
                clause = cov.get("clause")
                clause_norm = normalize_clause(clause)
                target_keys = template["answers"].get(scenario_id, {})
                matched_key = clause if clause in target_keys else next(
                    (k for k in target_keys if normalize_clause(k) == clause_norm), None
                )

                if matched_key is None: continue

                result = process_ledger(ledger_df, scenario_id, cov)
                template["answers"][scenario_id][matched_key] = result
                print(f"  [{scenario_id} | {matched_key}] -> {result['status']} | Actual: {result['actual']}")

    output_path = os.path.join(BASE_DIR, "submission.json")
    with open(output_path, "w", encoding="utf-8") as f: json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"\nГотово! Результат в {output_path}")

if __name__ == "__main__":
    main()