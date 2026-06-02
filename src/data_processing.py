"""Feature engineering and proxy target construction for credit risk modeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from category_encoders import WOEEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Sequence[str] = (
    "CustomerId",
    "Value",
    "Amount",
    "TransactionStartTime",
    "FraudResult",
    "ProductCategory",
    "ChannelId",
    "CurrencyCode",
)


def _safe_mode(s: pd.Series) -> str:
    m = s.mode(dropna=True)
    if len(m) == 0:
        return "unknown"
    return str(m.iloc[0])


def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw transactions to one row per customer (no RFM yet)."""
    df = transactions.copy()
    if "ProductId" not in df.columns:
        df["ProductId"] = "unknown"
    if "ProviderId" not in df.columns:
        df["ProviderId"] = "unknown"
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
    df["txn_hour"] = df["TransactionStartTime"].dt.hour
    df["txn_dow"] = df["TransactionStartTime"].dt.dayofweek

    grouped = df.groupby("CustomerId", observed=True)
    agg = grouped.agg(
        total_value=("Value", "sum"),
        avg_value=("Value", "mean"),
        std_value=("Value", "std"),
        min_value=("Value", "min"),
        max_value=("Value", "max"),
        txn_count=("Value", "count"),
        sum_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        std_amount=("Amount", "std"),
        fraud_rate=("FraudResult", "mean"),
        n_unique_products=("ProductId", pd.Series.nunique),
        n_unique_providers=("ProviderId", pd.Series.nunique),
        n_unique_channels=("ChannelId", pd.Series.nunique),
        avg_txn_hour=("txn_hour", "mean"),
        std_txn_hour=("txn_hour", "std"),
        avg_txn_dow=("txn_dow", "mean"),
    )

    cats = grouped.agg(
        dominant_product_category=("ProductCategory", _safe_mode),
        dominant_channel=("ChannelId", _safe_mode),
        dominant_currency=("CurrencyCode", _safe_mode),
    )

    out = pd.concat([agg, cats], axis=1)
    out["std_value"] = out["std_value"].fillna(0.0)
    out["std_amount"] = out["std_amount"].fillna(0.0)
    out["std_txn_hour"] = out["std_txn_hour"].fillna(0.0)
    return out.reset_index()


def compute_rfm(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute Recency (days), Frequency (count), Monetary (sum of Value) per customer."""
    t = transactions.copy()
    t["TransactionStartTime"] = pd.to_datetime(t["TransactionStartTime"], errors="coerce")
    last_txn = t.groupby("CustomerId", observed=True)["TransactionStartTime"].max()
    recency_days = (snapshot_date - last_txn).dt.days.astype(float)
    frequency = t.groupby("CustomerId", observed=True).size().astype(float)
    monetary = t.groupby("CustomerId", observed=True)["Value"].sum().astype(float)
    rfm = pd.DataFrame(
        {
            "CustomerId": recency_days.index.astype(int),
            "Recency": recency_days.values,
            "Frequency": frequency.reindex(recency_days.index).values,
            "Monetary": monetary.reindex(recency_days.index).values,
        }
    )
    return rfm


def merge_rfm_into_customers(customers: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    return customers.merge(rfm, on="CustomerId", how="left")


def identify_high_risk_cluster_from_frame(df: pd.DataFrame, cluster_col: str = "cluster") -> int:
    """Least engaged segment: lowest mean Frequency + Monetary among clusters."""
    stats = df.groupby(cluster_col, observed=True)[["Recency", "Frequency", "Monetary"]].mean()
    engagement = stats["Frequency"] + stats["Monetary"]
    high_risk = int(engagement.idxmin())
    logger.info("Cluster engagement stats:\n%s", stats.assign(engagement=engagement))
    return high_risk


def build_numeric_preprocess() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def expected_aggregate_columns() -> List[str]:
    """Columns produced after merge_rfm_into_customers (including RFM)."""
    return [
        "CustomerId",
        "total_value",
        "avg_value",
        "std_value",
        "min_value",
        "max_value",
        "txn_count",
        "sum_amount",
        "avg_amount",
        "std_amount",
        "fraud_rate",
        "n_unique_products",
        "n_unique_providers",
        "n_unique_channels",
        "avg_txn_hour",
        "std_txn_hour",
        "avg_txn_dow",
        "dominant_product_category",
        "dominant_channel",
        "dominant_currency",
        "Recency",
        "Frequency",
        "Monetary",
    ]


@dataclass
class CreditRiskDataPipeline(BaseEstimator, TransformerMixin):
    """Raw transactions -> model-ready frame with ``is_high_risk`` proxy target.

    Fit learns snapshot date, RFM KMeans segmentation, high-risk cluster id,
    WoE mappings, and numeric scaling. Transform applies the same artifacts.
    """

    random_state: int = 42
    n_clusters: int = 3
    categorical_features: tuple = (
        "dominant_product_category",
        "dominant_channel",
        "dominant_currency",
    )

    def __post_init__(self) -> None:
        self.snapshot_date_: Optional[pd.Timestamp] = None
        self._rfm_scaler = StandardScaler()
        self._kmeans: Optional[KMeans] = None
        self.high_risk_cluster_: Optional[int] = None
        self._woe: Optional[WOEEncoder] = None
        self._num_pipeline: Optional[Pipeline] = None
        self.numeric_features_: List[str] = []
        self.feature_columns_out_: List[str] = []

    def _build_customer_table(self, transactions: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in transactions.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.snapshot_date_ is None:
            ts = pd.to_datetime(transactions["TransactionStartTime"], errors="coerce")
            self.snapshot_date_ = pd.Timestamp(ts.max())

        cust = aggregate_transactions(transactions)
        rfm = compute_rfm(transactions, self.snapshot_date_)
        cust = merge_rfm_into_customers(cust, rfm)
        return cust

    def _rfm_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        rfm = np.log1p(df[["Recency", "Frequency", "Monetary"]].astype(float).values)
        return rfm

    def fit(self, transactions: pd.DataFrame, y=None):  # noqa: ARG002
        customers = self._build_customer_table(transactions)
        rfm_x = self._rfm_feature_matrix(customers)
        rfm_scaled = self._rfm_scaler.fit_transform(rfm_x)
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        customers = customers.copy()
        customers["cluster"] = self._kmeans.fit_predict(rfm_scaled)
        self.high_risk_cluster_ = identify_high_risk_cluster_from_frame(customers)
        customers["is_high_risk"] = (customers["cluster"] == self.high_risk_cluster_).astype(int)

        for col in self.categorical_features:
            customers[col] = customers[col].astype(str)

        self.numeric_features_ = [
            c
            for c in customers.columns
            if c
            not in (
                "CustomerId",
                "cluster",
                "is_high_risk",
                *self.categorical_features,
            )
            and customers[c].dtype != object
        ]

        self._woe = WOEEncoder(cols=list(self.categorical_features), random_state=self.random_state)
        self._woe.fit(
            customers[self.numeric_features_ + list(self.categorical_features)],
            customers["is_high_risk"],
        )

        woe_frame = self._woe.transform(
            customers[self.numeric_features_ + list(self.categorical_features)]
        )
        self._num_pipeline = build_numeric_preprocess()
        self._num_pipeline.fit(woe_frame[self.numeric_features_])

        woe_cols = [c for c in woe_frame.columns if c.endswith("_woe")]
        self.feature_columns_out_ = self.numeric_features_ + woe_cols
        self.customers_fit_ = customers
        return self

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if self.snapshot_date_ is None or self._kmeans is None or self.high_risk_cluster_ is None:
            raise RuntimeError("Call fit before transform.")
        customers = self._build_customer_table(transactions)
        rfm_scaled = self._rfm_scaler.transform(self._rfm_feature_matrix(customers))
        customers = customers.copy()
        customers["cluster"] = self._kmeans.predict(rfm_scaled)
        customers["is_high_risk"] = (customers["cluster"] == self.high_risk_cluster_).astype(int)
        for col in self.categorical_features:
            customers[col] = customers[col].astype(str)

        woe_frame = self._woe.transform(  # type: ignore[union-attr]
            customers[self.numeric_features_ + list(self.categorical_features)]
        )
        nums = self._num_pipeline.transform(  # type: ignore[union-attr]
            woe_frame[self.numeric_features_]
        )
        num_df = pd.DataFrame(nums, columns=self.numeric_features_, index=woe_frame.index)
        woe_only = woe_frame[[c for c in woe_frame.columns if c.endswith("_woe")]]
        out = pd.concat(
            [customers[["CustomerId", "is_high_risk", "cluster"]], num_df, woe_only],
            axis=1,
        )
        return out

    def fit_transform(self, transactions: pd.DataFrame, y=None, **fit_params):
        return self.fit(transactions, y=y, **fit_params).transform(transactions)


def read_transactions(path: str) -> pd.DataFrame:
    """Load raw CSV."""
    return pd.read_csv(path)
