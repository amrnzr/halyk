import pandas as pd

def process_ledger(csv_path: str, account_id: str, covenant: dict) -> dict:
    """
    Рассчитывает фактические показатели по транзакциям из CSV 
    для конкретного account_id и проверяет их на соблюдение ковенанта.
    """
    clause = covenant.get("clause")
    cov_type = covenant.get("type")
    limit = covenant.get("limit", 0.0)

    # 1. Загружаем CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Ошибка чтения файла выписки {csv_path}: {e}")
        return {"status": "UNKNOWN", "actual": 0.0, "evidence_txn_id": None}

    # Приводим названия колонок к нижнему регистру для предотвращения ошибок
    df.columns = [col.lower() for col in df.columns]

    # 2. ФИЛЬТРАЦИЯ: Берем транзакции ТОЛЬКО для нужного аккаунта
    account_col = 'account_id' if 'account_id' in df.columns else 'account'
    df_acc = df[df[account_col].astype(str).str.strip() == str(account_id).strip()]

    if df_acc.empty:
        return {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None}

    actual_value = 0.0
    status = "COMPLIANT"
    evidence_txn_id = None

    # Определяем имена колонок
    amount_col = 'amount' if 'amount' in df_acc.columns else 'sum'
    type_col = 'type' if 'type' in df_acc.columns else 'direction'
    txn_id_col = 'transaction_id' if 'transaction_id' in df_acc.columns else 'txn_id'

    # 3. ЛОГИКА РАСЧЕТА по типам ковенантов
    
    # ТИП 1: Максимальные расходы (max_expense)
    if cov_type == "max_expense":
        # Фильтруем дебетовые / расходные операции
        if type_col in df_acc.columns:
            expenses = df_acc[df_acc[type_col].str.upper().isin(['DEBIT', 'EXPENSE', 'OUT'])]
            actual_value = float(expenses[amount_col].sum())
        else:
            actual_value = float(df_acc[amount_col].sum())

        actual_value = round(actual_value, 2)
        if actual_value > limit:
            status = "BREACH"

    # ТИП 2: Минимальная выручка (min_revenue)
    elif cov_type == "min_revenue":
        if type_col in df_acc.columns:
            revenue = df_acc[df_acc[type_col].str.upper().isin(['CREDIT', 'REVENUE', 'IN'])]
            actual_value = float(revenue[amount_col].sum())
        else:
            actual_value = float(df_acc[amount_col].sum())

        actual_value = round(actual_value, 2)
        if actual_value < limit:
            status = "BREACH"

    # ТИП 3: Ограничение на разовую транзакцию (max_single_tx)
    elif cov_type == "max_single_tx":
        # Находим максимальную транзакцию
        actual_value = float(df_acc[amount_col].max()) if not df_acc.empty else 0.0
        actual_value = round(actual_value, 2)

        # Ищем транзакции, превышающие лимит
        breach_txs = df_acc[df_acc[amount_col] > limit]
        if not breach_txs.empty:
            status = "BREACH"
            # Извлекаем ID первой транзакции-нарушителя
            if txn_id_col in breach_txs.columns:
                evidence_txn_id = str(breach_txs.iloc[0][txn_id_col])

    return {
        "status": status,
        "actual": actual_value,
        "evidence_txn_id": evidence_txn_id
    }