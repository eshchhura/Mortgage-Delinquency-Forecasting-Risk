# Mortgage Delinquency Forecasting

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Model](https://img.shields.io/badge/Final%20Model-Oversampled%20LightGBM-green)
![Task](https://img.shields.io/badge/Task-Imbalanced%20Classification-purple)

A loan-level model that flags Fannie Mae single-family mortgages likely to become **90 or more days delinquent within a six-month window**, using a monthly loan-performance panel built from a 10% borrower sample of 2021 originations.

The goal is simple: given everything we know about a loan this month, can we predict whether it slides into serious delinquency soon enough for a servicing team to actually act on it?

---

## 1. Project Overview

A mortgage that quietly drifts toward delinquency is costly for the servicer, the investor, and the household alike, and by the time a loan is 90 days past due most of the good options are gone. This project builds a machine learning model that turns a large monthly loan panel into a short, ranked watchlist of loans most likely to default in the next six months.

The project focuses on:

- forecasting future serious (90+ day) delinquency at the loan-month level
- preserving time order so the model is scored on loans it never trained on
- handling extreme class imbalance (only ~0.25% of loan-months are positive)
- comparing linear, tree-based, and neural models on the right metrics
- choosing an operating threshold that produces a watchlist a team can work
- validating the chosen threshold out-of-time and checking score calibration

The final model is an **oversampled LightGBM classifier** with a 0.70 decision threshold, scored once on a held-out, time-separated test set.

---

## 2. Business Problem

Serious delinquency is rare but expensive, and it is far easier to manage early than late. This project is framed for a mortgage **servicing or credit-risk team** that needs a forward-looking screen to support decisions such as:

- which borrowers to call first
- where to target loss-mitigation outreach
- how to size reserves against the book
- which loans to flag for closer monitoring before they go bad

The main modeling question:

> Given everything we know about a loan this month, can we predict whether it becomes 90+ days delinquent within the next six months?

The audience cares about three things: catching as many future defaults as possible, keeping the resulting watchlist small enough to work, and trusting that the model was scored honestly on loans from a period it was never trained on.

---

## 3. Data

The base dataset is **Fannie Mae's public Single-Family Loan Performance** dataset, the standard benchmark for U.S. mortgage credit research, built from its acquisition and performance files.

Dataset link: [Fannie Mae Single-Family Loan Performance Data](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data)

Raw and modeling scale:

| Item | Value |
|---|---:|
| Source loan-month rows | ~13.9 million |
| Unique loans (10% sample) | 272,990 |
| Origination window | 2021 Q1–Q2 |
| Panel time span | Jan 2021 – Nov 2025 |
| Panel columns | 68 |
| Final modeling rows | 12,524,939 labeled loan-months |

Each row is **one loan in one reporting month**: its current balance and delinquency status, joined to fixed origination facts such as credit scores, DTI, LTV/CLTV, interest rate, loan purpose, occupancy, property type, and geography. Sampling is done **by loan** (a deterministic 10% hash), so every kept loan retains its complete month-by-month timeline and the forward-looking label stays valid.

---

## 4. Repository Structure

```text
Mortgage-Delinquency-Forecasting/
│
├── data/
│   └── processed/
│       ├── loan_month_panel_2021q1_q2_sample10pct.parquet
│       └── preprocessing_outputs_2021q1_q2_sample10pct/
│           ├── X_train_processed.npz
│           ├── X_test_processed.npz
│           ├── y_train.csv
│           ├── y_test.csv
│           ├── feature_names.csv
│           ├── preprocessing_summary.csv
│           ├── test_meta.csv
│           └── preprocessor.joblib
│
├── models/
│   └── 2021q1_q2_sample10pct/
│       ├── final_oversampled_lgbm_model.joblib
│       ├── final_model_metrics.csv
│       ├── model_comparison.csv
│       └── feature_importance.csv
│
├── Notebooks/
│   ├── Risk_Data_Wrangling.ipynb          # initial single-file cleaning pass
│   ├── Risk_Data_Wrangling_Revised.ipynb  # production per-quarter panel builder
│   ├── Risk_EDA.ipynb
│   ├── Risk_Preprocessing.ipynb
│   └── Risk_Modeling.ipynb
│
├── Reports/
│   ├── Mortgage-Delinquency-Report.pdf
│   └── Mortgage_Delinquency_Forecasting.pptx
│
└── README.md
```

---

## 5. Project Workflow

### Data Wrangling

The raw files are large, so wrangling is per-quarter, chunked, and memory-conscious. Because Fannie Mae organizes performance files by acquisition quarter, each loan's full timeline lives in a single file, so labeling can be done per file and the panels concatenated afterward.

Main steps:

- streamed each raw file in chunks, cleaning, typing, and downcasting each chunk before reading the next
- kept a deterministic, hash-based 10% sample **by loan** so every kept loan keeps its full history
- joined acquisition and performance into one row per loan-month, with origination terms aligned to each loan
- parsed dates once and renamed fields for readability
- engineered delinquency history: rolling 12-month counts of 30+ and 60+ day past-due spells, prior status, and status-change flags
- built the forward label by looking six months ahead
- dropped the unlabeled tail (rows without a full six-month forward window) so every label is real and complete

The modeling target is:

```text
future_serious_dq_6m
```

at the loan-month level (presented as `future_90plus_dq_6m` in the written report). A row is 1 if the loan reaches 90+ days delinquent at any point in the next six reporting months, otherwise 0.

### Exploratory Data Analysis

All rates below are at the loan-month grain. The full-sample positive rate is about **0.25%** (roughly one row in 400), rising to 0.30% in the later, more-seasoned test window.

#### 1. Credit score sets the baseline

The 90+ day delinquency rate falls **more than thirty-fold** across origination credit-score deciles, from 1.01% in the lowest band to 0.03% in the highest, monotonically and in exactly the direction underwriting intuition expects. It is the single strongest static signal in the data.

#### 2. Recent lateness is the loudest near-term signal

A loan's current delinquency status dwarfs every fixed origination feature when predicting the next six months:

| Current status | 6-month 90+ DQ rate |
|---|---:|
| Current | 0.11% |
| 30 days late | 13.0% |
| 60 days late | 55.9% |

#### 3. Leverage and origination channel

Higher debt-to-income climbs steadily with delinquency, the mirror image of the credit-score curve. Origination channel matters too, with retail-originated loans running noticeably safer than broker and correspondent channels:

| Channel | 90+ DQ rate |
|---|---:|
| Correspondent | 0.32% |
| Broker | 0.29% |
| Retail | 0.21% |

---

## 6. Feature Engineering

The final model uses origination terms, balance and paydown variables, engineered delinquency history, and time context.

| Feature group | Features |
|---|---|
| Origination terms | borrower & co-borrower credit scores, `dti`, `original_ltv` / `original_cltv`, interest rate, loan purpose, occupancy, property type |
| Balance & paydown | `original_upb`, `current_actual_upb`, `upb_ratio`, `paydown_pct`, `total_principal_current` |
| Delinquency history | rolling 12-month 30+ / 60+ counts, prior status, status-change flags, current status |
| Time context | `loan_age`, months remaining to maturity, reporting month / quarter / year |

The pipeline one-hot encodes 14 categorical fields and standard-scales 32 numeric fields into a **172-column** sparse design matrix. Trees don't need the scaling, but the pipeline applies it uniformly so every model is compared on identical inputs. Every history and label feature is computed using only information available as of the reporting month, so no future information leaks into a row.

---

## 7. Train/Test Strategy

This is a forecasting problem, so the test set must live in the future relative to training. A random split would let the model peek at later months while scoring earlier ones, inflating every metric.

The setup:

- chronological 80/20 train/test split
- **Train:** 10,059,227 loan-months (Jan 2021 – Jul 2024)
- **Test:** 2,465,712 loan-months (Aug 2024 – Jun 2025)
- model selection, hyperparameter tuning, and threshold selection all happen inside the training window, with the latest training months held out as a validation set
- the test set is scored once, after every decision is locked
- an isotonic calibration check maps the oversampled model's inflated scores back onto true probabilities, ready if the scores ever need to feed expected-loss math

---

## 8. Models Tested

I compared a baseline, linear models, boosted trees, and a neural network, with both class-weighting and random-oversampling treatments for the imbalance.

Models tested:

- Dummy (predict majority class)
- Logistic Regression (class-weighted and oversampled)
- LightGBM (class-weighted, tuned, and oversampled)
- XGBoost (class-weighted and tuned)
- Feedforward Neural Network (MLP)

Because serious delinquency is rare, models are judged on PR-AUC, recall, and precision rather than accuracy, and oversampling is applied to the training data only so the test set still reflects the real class balance.

### Model Comparison (held-out test, 0.5 cutoff)

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---:|---:|---:|---:|
| **Oversampled LightGBM (final)** | **0.949** | **0.340** | **0.75** | **0.121** |
| Tuned XGBoost | 0.948 | 0.337 | 0.82 | 0.044 |
| Neural Network | 0.941 | 0.329 | 0.26 | 0.534 |
| Oversampled Logistic Regression | 0.942 | 0.224 | 0.72 | 0.132 |
| Logistic Regression | 0.943 | 0.213 | 0.82 | 0.034 |
| Dummy (predict majority) | 0.500 | 0.003 | 0.00 | 0.00 |

ROC-AUC clusters tightly between 0.90 and 0.95 and can't separate the models; PR-AUC can, and the gradient-boosted trees pull ahead. Random oversampling (capping the positive class at 20% of training rows) turned out to handle the imbalance better than a heavy class weight for LightGBM.

---

## 9. Final Model Performance

The final model, oversampled LightGBM at a 0.70 decision threshold, was evaluated once on the untouched chronological holdout.

| Metric | Final test score |
|---|---:|
| ROC-AUC | 0.949 |
| PR-AUC | 0.340 |
| Recall | 0.71 |
| Precision | 0.157 |
| Flag rate | 1.36% |

Confusion matrix on 2,465,712 test loan-months:

| | Predicted: no DQ | Predicted: 90+ DQ |
|---|---:|---:|
| **Actual: no DQ** | 2,430,072 | 28,255 |
| **Actual: 90+ DQ** | 2,113 | 5,272 |

The model catches **71% of the loans** that go on to serious delinquency while flagging just **1.36% of the book**. About one flagged loan in six is a true future default, roughly fifty times the 0.3% base rate, which is the kind of lift that makes an outreach queue worth working.

The most-used features echo the EDA: borrower credit score, DTI, loan size (original and current UPB), co-borrower credit score, and rate/LTV terms. No single exotic feature carries the model; it is familiar credit fundamentals weighed together at scale. (Reported importances are LightGBM split counts, which favor high-cardinality numerics and understate the binary delinquency-status flags; a gain-based or SHAP ranking would lift those flags higher.)

---

## 10. Key Takeaways

- This is a rare-event problem: accuracy is meaningless, since a "never delinquent" model scores 99.7% and catches nothing.
- Credit fundamentals dominate, but **recent delinquency status** is by far the loudest near-term signal.
- Random oversampling beat heavy class-weighting for the boosted trees.
- A time-ordered split with a validation-chosen threshold is essential; a random split would overstate every metric.
- A recall-first operating point turns 12.5 million loan-months into a short, workable watchlist that stays explainable to risk teams.

---

## 11. How to Reproduce

Clone the repository:

```bash
git clone https://github.com/eshchhura/Mortgage-Delinquency-Forecasting.git
cd Mortgage-Delinquency-Forecasting
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

Install the main Python packages:

```bash
pip install pandas numpy scipy scikit-learn lightgbm xgboost imbalanced-learn matplotlib joblib pyarrow jupyter
```

Download the Fannie Mae Single-Family Loan Performance files and place the raw data in the expected local location used by the notebooks, or update the file paths in the notebooks.

Recommended notebook order:

1. `Notebooks/Risk_Data_Wrangling_Revised.ipynb`
2. `Notebooks/Risk_EDA.ipynb`
3. `Notebooks/Risk_Preprocessing.ipynb`
4. `Notebooks/Risk_Modeling.ipynb`

---

## 12. Limitations and Future Improvements

This model performs well using loan-level and recent-history signals, but future versions could improve by adding context and sharpening the outputs.

Potential improvements:

- calibrate probabilities (isotonic) so scores can feed reserve sizing, not just ranking
- add macro context such as rates and regional unemployment to harden against portfolio shift
- tie the decision threshold to outreach capacity rather than a fixed recall target
- monitor drift and set a retrain cadence as newer origination vintages season
- use gain-based or SHAP importance to better surface the delinquency-history features
- extend beyond the 2021 H1 origination cohort to test stability across vintages

---

## Author

**Esh S. Chhura**
GitHub: [@eshchhura](https://github.com/eshchhura)
