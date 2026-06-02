"""Unit tests for data processing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing import (
    CreditRiskDataPipeline,
    aggregate_transactions,
    compute_rfm,
    expected_aggregate_columns,
    merge_rfm_into_customers,
)


def _synthetic_transactions(n_customers: int = 30, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(42)
    rows = []
    base = pd.Timestamp("2018-11-01")
    for cid in range(1, n_customers + 1):
        n_tx = int(rng.integers(3, 25))
        for _ in range(n_tx):
            t = base + pd.Timedelta(days=int(rng.integers(0, 60)), hours=int(rng.integers(0, 23)))
            rows.append(
                {
                    "CustomerId": cid,
                    "Value": float(rng.integers(500, 5000)),
                    "Amount": float(rng.integers(500, 5000)),
                    "TransactionStartTime": t.isoformat(),
                    "FraudResult": int(rng.integers(0, 2)),
                    "ProductCategory": ["airtime", "data", "utility"][int(rng.integers(0, 3))],
                    "ChannelId": int(rng.integers(1, 4)),
                    "CurrencyCode": 256,
                    "ProductId": f"P{rng.integers(1, 5)}",
                    "ProviderId": f"PR{rng.integers(1, 4)}",
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_transactions_columns():
    df = _synthetic_transactions(10)
    agg = aggregate_transactions(df)
    rfm = compute_rfm(df, pd.Timestamp(df["TransactionStartTime"].max()))
    out = merge_rfm_into_customers(agg, rfm)
    expected = expected_aggregate_columns()
    for col in expected:
        assert col in out.columns, f"missing {col}"


def test_credit_risk_pipeline_fit_transform():
    df = _synthetic_transactions(40, rng=np.random.default_rng(0))
    pipe = CreditRiskDataPipeline(random_state=7)
    pipe.fit(df)
    transformed = pipe.transform(df)
    assert "is_high_risk" in transformed.columns
    assert transformed["is_high_risk"].isin([0, 1]).all()
    assert pipe.high_risk_cluster_ is not None
