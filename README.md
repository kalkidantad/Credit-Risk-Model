# Credit Risk Probability Model (Alternative Data)

End-to-end credit risk modeling for Bati Bank’s buy-now-pay-later partnership using eCommerce transaction data (Xente-style schema). The project covers business framing under Basel II, exploratory analysis, feature engineering with WoE/IV-style encoding, an RFM-based proxy for default risk, model training with MLflow, and a containerized FastAPI scoring service.

## Repository layout

```
credit-risk-model/
├── .github/workflows/ci.yml
├── data/raw/                 # Place downloaded CSV here (gitignored)
├── data/processed/           # Pipeline outputs (gitignored)
├── notebooks/eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick start

1. **Data**  
   Download the [Xente Challenge](https://www.kaggle.com/datasets) dataset and save it as `data/raw/Xente_challenge_dataset.csv` (or set `CREDIT_RISK_RAW_DATA` to your file path).

2. **Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Build processed dataset and train (MLflow)**

   ```bash
   export MLFLOW_TRACKING_URI=./mlruns
   python -m src.train --raw-path data/raw/Xente_challenge_dataset.csv --register-model
   ```

4. **API (local)**

   ```bash
   export MLFLOW_TRACKING_URI=./mlruns
   export MLFLOW_MODEL_URI="models:/credit_risk_classifier/Production"
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

5. **Docker**

   ```bash
   docker compose up --build
   ```

## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence the need for an interpretable and well-documented model?

Basel II expects banks to hold capital in line with measured credit risk and to demonstrate that risk estimates are **accurate, stable, and auditable**. That shifts modeling from “best leaderboard score” to a **governed production artifact**: data definitions, target definition, assumptions, validation evidence, and ongoing monitoring must be traceable for internal audit and supervisory review. Interpretability (for example, clear drivers of risk and controlled monotonicity where appropriate) and documentation are therefore not optional polish; they are how the institution proves the model is **fit for use** in capital and credit decisions and how it detects drift or degradation before losses materialize.

### Without a direct “default” label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

Retail BNPL performance labels (charge-off, 90+ days past due) are often **absent in partner data** or delayed relative to onboarding needs. A proxy (here, behavioral disengagement via RFM segmentation) provides a **timely, automatable target** so supervised models can be trained on observable behavior. The business risk is structural: the proxy is **not repayment outcome**. It can confuse “low platform usage” with “inability to repay,” embed partner-specific seasonality, and bake in **concept drift** if engagement patterns change. Decisions must be framed as **ranking and triage under uncertainty**, with policy limits, human review for edge cases, and relabeling when true outcomes become available.

### What are the key trade-offs between a simple, interpretable model (for example, logistic regression with WoE) and a high-performance model (for example, gradient boosting) in a regulated financial context?

| Dimension | Simple / interpretable (e.g., LR + WoE) | High-performance (e.g., boosting) |
|-----------|----------------------------------------|-----------------------------------|
| Governance | Easier to document monotonic risk drivers and segment-level behavior; aligns with classic scorecards. | Stronger fit to nonlinearities; harder to explain without additional tooling (SHAP, constrained trees). |
| Stability | Often smoother under small sample shifts when regularized and binned. | Can chase noise if not tuned/regularized; may need stronger monitoring. |
| Performance ceiling | May leave signal on the table in complex behavioral data. | Often better discrimination when validated rigorously. |
| Validation burden | Lower for linear, binned structures if assumptions hold. | Higher: leakage checks, calibration, fairness slices, stress tests. |

In practice, regulated teams often **start with a transparent baseline**, compare against a **constrained** high-capacity model, and pick based on **out-of-time validation**, documentation cost, and monitoring feasibility—not raw accuracy alone.

## Methodology (summary)

- **Proxy target**: Customer-level RFM at a fixed snapshot; K-Means (`k=3`, scaled RFM, fixed `random_state`) identifies a low-engagement segment; that cluster is labeled `is_high_risk=1` (see `src/data_processing.py`).
- **Features**: Aggregates (sums, counts, dispersion), simple fraud mix, and temporal summaries from `TransactionStartTime`; categoricals encoded with **Weight of Evidence** using the proxy target to align with credit-scoring practice.
- **Training**: Stratified holdout by customer, multiple estimators, optional hyperparameter search, MLflow logging and model registry.
- **Serving**: FastAPI loads the registered model by URI (configurable).

## Limitations

The proxy is a **modeling assumption**, not ground-truth default. Results should not be interpreted as literal probability of default without calibration to labeled outcomes. Partner data may not represent the bank’s full applicant population.

## Team

Kerod, Mahbubah, Feven — 10 Academy AI Mastery, Week 4.

## License

Educational use for the 10 Academy challenge.
