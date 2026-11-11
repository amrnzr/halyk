import pandas as pd

DEFAULT_FX_TO_USD = {
    "USD": 1.0,
    "EUR": 1.08,
}

def _row_to_usd(row, amount_col, currency_col, fx_table):
    if currency_col is None:
        return row[amount_col]
    ccy = str(row[currency_col]).strip().upper()
    rate = fx_table.get(ccy)
    if rate is None:
        print(f"⚠️ Неизвестная валюта '{ccy}' у транзакции {row.get('txn_id', '?')} "
              f"— сумма НЕ сконвертирована, добавьте курс в FX_TO_USD")
        return row[amount_col]
    return row[amount_col] * rate

def load_ledger(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    return df

def process_ledger(df: pd.DataFrame, scenario_id: str, covenant: dict, fx_table: dict = None) -> dict:
    fx_table = fx_table or DEFAULT_FX_TO_USD
    clause = covenant.get("clause")
    cov_type = covenant.get("type")
    limit = float(covenant.get("limit", 0.0))
    period_start = covenant.get("period_start")
    period_end = covenant.get("period_end")

    cols = {c.lower(): c for c in df.columns}
    txn_id_col = cols.get("txn_id") or cols.get("transaction_id")
    amount_col = cols.get("amount") or cols.get("sum")
    type_col = cols.get("type") or cols.get("direction")
    currency_col = cols.get("currency") or cols.get("ccy")
    date_col = cols.get("date") or cols.get("txn_date")

    if txn_id_col is None or amount_col is None:
        print(f"⚠️ Не найдены обязательные колонки (txn_id/amount) для {scenario_id}/{clause}")
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    prefix = f"TXN-{scenario_id}-".upper()
    df_acc = df[df[txn_id_col].astype(str).str.upper().str.startswith(prefix)].copy()

    if df_acc.empty:
        print(f"⚠️ Нет транзакций с префиксом {prefix} — проверьте сопоставление scenario_id (пункт {clause})")
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    missing_amount = df_acc[df_acc[amount_col].isnull()]
    if not missing_amount.empty:
        ids = missing_amount[txn_id_col].tolist()
        print(f"⚠️ У {scenario_id} есть транзакции без суммы: {ids}. Эти строки исключены из расчёта.")
        df_acc = df_acc[df_acc[amount_col].notnull()]

    if period_start or period_end and date_col:
        if date_col:
            mask = pd.Series(True, index=df_acc.index)
            if period_start:
                mask &= df_acc[date_col] >= period_start
            if period_end:
                mask &= df_acc[date_col] <= period_end
            df_acc = df_acc[mask]

    if df_acc.empty:
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    df_acc["_usd"] = df_acc.apply(lambda r: _row_to_usd(r, amount_col, currency_col, fx_table), axis=1)

    status = "COMPLIANT"
    actual_value = 0.0
    evidence_txn_id = None

    if cov_type == "external_metric":
        print(f"ℹ️ {scenario_id}: пункт {clause} ссылается на материнскую компанию (внешняя метрика). Леджер заемщика здесь неприменим -> UNKNOWN")
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    elif cov_type == "max_expense":
        if type_col:
            rows = df_acc[df_acc[type_col].astype(str).str.upper().isin(["DEBIT", "EXPENSE", "OUT"])]
        else:
            rows = df_acc[df_acc["_usd"] < 0]
        actual_value = abs(round(float(rows["_usd"].sum()), 2))
        status = "BREACH" if actual_value > limit else "COMPLIANT"

    elif cov_type == "min_revenue":
        if type_col:
            rows = df_acc[df_acc[type_col].astype(str).str.upper().isin(["CREDIT", "REVENUE", "IN"])]
        else:
            rows = df_acc[df_acc["_usd"] > 0]
        actual_value = abs(round(float(rows["_usd"].sum()), 2))
        status = "BREACH" if actual_value < limit else "COMPLIANT"

    elif cov_type == "min_quarterly_revenue":
        if date_col is None:
            return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}
        rows = df_acc[df_acc[type_col].astype(str).str.upper().isin(["CREDIT", "REVENUE", "IN"])].copy() if type_col else df_acc[df_acc["_usd"] > 0].copy()
        rows["_q"] = pd.to_datetime(rows[date_col]).dt.to_period("Q")
        by_quarter = rows.groupby("_q")["_usd"].sum()
        if period_start: by_quarter = by_quarter[by_quarter.index >= pd.Period(period_start, freq="Q")]
        if period_end: by_quarter = by_quarter[by_quarter.index <= pd.Period(period_end, freq="Q")]
        if by_quarter.empty: return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}
        
        actual_value = round(abs(float(by_quarter.min())), 2)
        status = "BREACH" if by_quarter.min() < limit else "COMPLIANT"
        return {"status": status, "actual": actual_value, "evidence_txn_id": None}

    elif cov_type == "max_single_tx":
        df_acc["_abs"] = df_acc["_usd"].abs()
        idx_max = df_acc["_abs"].idxmax()
        actual_value = round(float(df_acc.loc[idx_max, "_abs"]), 2)
        breach_txs = df_acc[df_acc["_abs"] > limit]
        if not breach_txs.empty:
            status = "BREACH"
            worst_idx = breach_txs["_abs"].idxmax()
            evidence_txn_id = str(breach_txs.loc[worst_idx, txn_id_col])

    elif cov_type == "min_balance":
        sort_col = date_col if date_col else txn_id_col
        df_sorted = df_acc.sort_values(sort_col)
        running = df_sorted["_usd"].cumsum()
        actual_value = round(float(running.min()), 2)
        status = "BREACH" if actual_value < limit else "COMPLIANT"
        return {"status": status, "actual": round(actual_value, 2), "evidence_txn_id": None}

    else:
        print(f"⚠️ Неизвестный тип ковенанта '{cov_type}' (пункт {clause}) -> UNKNOWN")
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    return {"status": status, "actual": round(abs(actual_value), 2), "evidence_txn_id": evidence_txn_id}