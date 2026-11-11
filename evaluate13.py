import pandas as pd

# --------------------------------------------------------------------------
# Курсы валют к USD. В датасете упоминаются "разные валюты" в леджере — их
# нужно свести к одной валюте (USD, судя по формату submission_template.json)
# прежде чем суммировать. Реальные курсы, скорее всего, указаны в одном из
# документов датасета (аудиторский отчёт / примечания к отчётности) — найдите
# их и впишите сюда. Пока здесь только заглушка для USD, чтобы код не падал,
# если в леджере всего одна валюта.
# --------------------------------------------------------------------------
DEFAULT_FX_TO_USD = {
    "USD": 1.0,
    "EUR": 1.08,
    # ФАКТ ПО ВАШЕМУ ЛЕДЖЕРУ: валюты в датасете — только USD и EUR, и EUR
    # встречается в КАЖДОМ из 27 сценариев (в среднем ~4% транзакций,
    # некоторые суммы в десятки миллионов). Без курса эти суммы либо
    # игнорируются (см. предупреждение в консоли), либо считаются 1:1 —
    # оба варианта, скорее всего, дадут неверный actual. Найдите официальный
    # курс EUR->USD в документах (ищите "notes to accounts", "учётная
    # политика", "FX rate", "курс валют") и впишите его сюда, например:
    # "EUR": 1.08,
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
    """Читает леджер один раз. Вызывайте это в main.py, а не внутри process_ledger."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    return df


def process_ledger(df: pd.DataFrame, scenario_id: str, covenant: dict,
                    fx_table: dict = None) -> dict:
    """
    Рассчитывает фактический показатель и статус ковенанта для scenario_id.

    ВАЖНО: фильтрация транзакций идёт по префиксу txn_id ("TXN-{scenario_id}-"),
    а не по колонке account_id — так задаётся связь заёмщик<->транзакция
    в условиях конкурса, и это не зависит от того, насколько точно LLM
    извлекла account_id из PDF.

    covenant может (опционально) содержать 'period_start' и 'period_end'
    (строки 'YYYY-MM-DD') — если в договоре ковенант привязан к периоду
    (квартал/год), а не ко всей истории леджера (в вашем датасете леджер
    покрывает 13+ месяцев, так что это может быть важно).
    """
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
        print(f"⚠️ Нет транзакций с префиксом {prefix} — проверьте сопоставление scenario_id "
              f"(пункт {clause})")
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    # ГРЯЗНЫЕ ДАННЫЕ: строки с пустой суммой встречаются в реальном датасете
    # (минимум 2 такие транзакции подтверждены в вашем леджере). Молча
    # суммировать NaN как 0 рискованно — это может быть именно та
    # транзакция, что определяет breach. Явно предупреждаем и исключаем её
    # из расчёта, а не роняем скрипт.
    missing_amount = df_acc[df_acc[amount_col].isnull()]
    if not missing_amount.empty:
        ids = missing_amount[txn_id_col].tolist()
        print(f"⚠️ У {scenario_id} есть транзакции без суммы: {ids} — "
              f"ПРОВЕРЬТЕ ВРУЧНУЮ по исходному документу, сумма могла быть "
              f"искажена намеренно ('грязные данные'). Эти строки исключены из расчёта.")
        df_acc = df_acc[df_acc[amount_col].notnull()]

    if period_start or period_end and date_col:
        if date_col:
            mask = pd.Series(True, index=df_acc.index)
            if period_start:
                mask &= df_acc[date_col] >= period_start
            if period_end:
                mask &= df_acc[date_col] <= period_end
            df_acc = df_acc[mask]
        else:
            print(f"⚠️ Ковенант {clause} задаёт период, но в леджере нет колонки даты")

    if df_acc.empty:
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    df_acc["_usd"] = df_acc.apply(
        lambda r: _row_to_usd(r, amount_col, currency_col, fx_table), axis=1
    )

    status = "COMPLIANT"
    actual_value = 0.0
    evidence_txn_id = None

    if cov_type == "max_expense":
        if type_col:
            rows = df_acc[df_acc[type_col].astype(str).str.upper().isin(
                ["DEBIT", "EXPENSE", "OUT"])]
        else:
            rows = df_acc[df_acc["_usd"] < 0]
        actual_value = abs(round(float(rows["_usd"].sum()), 2))
        status = "BREACH" if actual_value > limit else "COMPLIANT"

    elif cov_type == "min_revenue":
        if type_col:
            rows = df_acc[df_acc[type_col].astype(str).str.upper().isin(
                ["CREDIT", "REVENUE", "IN"])]
        else:
            rows = df_acc[df_acc["_usd"] > 0]
        actual_value = abs(round(float(rows["_usd"].sum()), 2))
        status = "BREACH" if actual_value < limit else "COMPLIANT"

    elif cov_type == "min_quarterly_revenue":
        # Ковенант тестируется ПОКВАРТАЛЬНО — соблюдение за год не покрывает
        # нарушение в отдельном квартале (см. формулировку "выполнение
        # показателя за год не устраняет его нарушение в отдельном квартале").
        if date_col is None:
            print(f"⚠️ min_quarterly_revenue требует колонку даты, которой нет для {scenario_id}")
            return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

        if type_col:
            rows = df_acc[df_acc[type_col].astype(str).str.upper().isin(
                ["CREDIT", "REVENUE", "IN"])].copy()
        else:
            rows = df_acc[df_acc["_usd"] > 0].copy()

        rows["_q"] = pd.to_datetime(rows[date_col]).dt.to_period("Q")
        by_quarter = rows.groupby("_q")["_usd"].sum()

        if period_start:
            start_q = pd.Period(period_start, freq="Q")
            by_quarter = by_quarter[by_quarter.index >= start_q]
        if period_end:
            end_q = pd.Period(period_end, freq="Q")
            by_quarter = by_quarter[by_quarter.index <= end_q]

        if by_quarter.empty:
            return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

        worst_quarter = by_quarter.idxmin()
        actual_value = round(abs(float(by_quarter.min())), 2)
        status = "BREACH" if by_quarter.min() < limit else "COMPLIANT"
        print(f"ℹ️ {scenario_id}: худший квартал по выручке — {worst_quarter} "
              f"(${by_quarter.min():,.2f})")
        return {
            "status": status,
            "actual": actual_value,
            "evidence_txn_id": None,  # агрегатный поквартальный тест -> null
        }

    elif cov_type == "max_single_tx":
        # ФИКС: раньше сравнивалось "amount > limit", что пропускало расходы,
        # записанные отрицательными числами. Теперь сравниваем по модулю.
        df_acc["_abs"] = df_acc["_usd"].abs()
        idx_max = df_acc["_abs"].idxmax()
        actual_value = round(float(df_acc.loc[idx_max, "_abs"]), 2)

        breach_txs = df_acc[df_acc["_abs"] > limit]
        if not breach_txs.empty:
            status = "BREACH"
            worst_idx = breach_txs["_abs"].idxmax()
            evidence_txn_id = str(breach_txs.loc[worst_idx, txn_id_col])
        else:
            status = "COMPLIANT"

    elif cov_type == "min_balance":
        # Минимальный остаток нарастающим итогом за период.
        sort_col = date_col if date_col else txn_id_col
        df_sorted = df_acc.sort_values(sort_col)
        running = df_sorted["_usd"].cumsum()
        actual_value = round(float(running.min()), 2)
        status = "BREACH" if actual_value < limit else "COMPLIANT"
        return {
            "status": status,
            "actual": round(actual_value, 2),  # остаток может быть отрицательным по смыслу
            "evidence_txn_id": None,           # это агрегатный тест -> null по правилам
        }

    else:
        # Раньше неизвестный тип молча проваливался в default COMPLIANT/0.0.
        # Теперь он явно помечается UNKNOWN, чтобы вы не пропустили ковенанты
        # (ratio-тесты, carve-outs и т.д.), которые эта функция пока не умеет считать.
        print(f"⚠️ Неизвестный/неподдерживаемый тип ковенанта '{cov_type}' "
              f"(пункт {clause}) — заполните эту ячейку вручную")
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    return {
        "status": status,
        "actual": round(abs(actual_value), 2),
        "evidence_txn_id": evidence_txn_id,
    }
