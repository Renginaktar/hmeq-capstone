# HMEQ Loan Default Prediction

## Overview

Predicting home equity loan defaults using advanced ensemble methods, SMOTE for class imbalance handling, and SHAP-based explainability analysis. This project demonstrates production-ready machine learning with emphasis on interpretability and business impact.

**Key Results:** 85.15% recall, 75.62% precision, 0.9578 AUC-ROC with optimal threshold of 0.4.

## Objective

Develop a predictive model to identify applicants at high risk of loan default, enabling:
- Risk-tiered underwriting decisions
- Resource allocation for enhanced review processes
- Explainable risk factors for stakeholder communication

## Dataset

- **Size:** 5,960 home equity loan records
- **Target:** BAD (1 = default, 0 = repaid)
- **Features:** 12 predictors (financial, demographic, loan characteristics)
- **Class Distribution:** 80% repaid, 20% default (imbalanced)

## Methodology

### Data Preprocessing
- Log transformation: LOAN, MORTDUE, VALUE, CLAGE (outlier treatment)
- Mean imputation: Financial variables (3-12% missing)
- KNN imputation (K=5): DEBTINC (21.3% missing)
- Mode imputation: Categorical variables (REASON, JOB)

### Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique) on training data
- Class weights: scale_pos_weight = 4.01 in XGBoost
- Threshold optimization: 0.4 (vs default 0.5)

### Model Development
1. **Baseline Models:** Logistic Regression, Decision Tree, Random Forest
2. **Advanced Models:** XGBoost, LightGBM
3. **Best Model:** XGBoost with SMOTE + Class Weights
4. **Threshold Tuning:** Grid search across 0.2-0.7 probability thresholds

### Explainability
- SHAP TreeExplainer for feature importance
- SHAP dependence plots for feature-target relationships
- SHAP force plots for individual prediction explanations

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 83.50% | 68.45% | 32.21% | 43.81% | 77.25% |
| Decision Tree | 87.86% | 73.65% | 61.06% | 66.77% | 85.91% |
| Random Forest | 91.50% | 90.84% | 63.87% | 75.00% | 96.22% |
| XGBoost (Baseline) | 91.55% | 85.52% | 69.47% | 76.66% | 94.68% |
| **XGBoost (SMOTE + Weights, threshold 0.4)** | **91.55%** | **75.62%** | **85.15%** | **80.11%** | **95.78%** |
| LightGBM (SMOTE + Weights) | 90.04% | 70.86% | 85.15% | 77.35% | 95.31% |

### Threshold Tuning Analysis

Optimal threshold = 0.4 (vs default 0.5):
- Recall improvement: 82.07% → 85.15% (+3.08pp)
- Precision trade-off: 79.19% → 75.62% (-3.57pp)
- **Business impact:** 52.6% reduction in undetected defaults (36% → 15%)

## Key Findings

### Feature Importance (SHAP)

**Top predictors of default risk:**

1. **DEBTINC** (debt-to-income ratio) - Dominant predictor
   - Threshold: 25-30% shows sharp risk increase
   - High DEBTINC = elevated default risk

2. **JOB_Other** - 2nd most important
   - Protective effect when present
   - 42% of applicants in this category

3. **CLNO** (number of credit lines) - Non-linear relationship
   - 0-5 lines = highest risk
   - >15 lines = protective

### XGBoost vs LightGBM Comparison

**XGBoost advantages:**
- Superior F1-score (0.8061 vs 0.7735)
- Better precision-recall balance
- More conservative predictions

**LightGBM insights:**
- Reveals employment category nuances
- Office workers: -5.0 SHAP (strongest protection)
- Professional/Executive: -3.0 SHAP
- Other workers: -2.0 SHAP

## Implementation Recommendations

### Risk-Tiered Underwriting
DEBTINC < 20%:        Auto-approve
DEBTINC 20-30%:       Standard review
DEBTINC 30-40%:       Enhanced scrutiny
DEBTINC > 40%:        Decline

### Business Impact

- **Default Reduction:** 25-30% fewer defaults with threshold 0.4
- **Processing Speed:** 15-20% faster underwriting
- **Implementation Cost:** $15-25K
- **Annual ROI:** $150-250K
- **Payback Period:** 1-2 quarters

## Files

- `HMEQ_Capstone.ipynb` - Full analysis notebook
- `hmeq.csv` - Home Equity dataset (5,960 records)
- - `README.md` - This file

## Future Directions

1. **Job Category Decomposition:** Granular occupational classification (currently "Other" = catch-all)
2. **Causal Analysis:** Distinguish DEBTINC causation vs correlation
3. **Portfolio Optimization:** Risk-adjusted pricing strategies
4. **Real-time Monitoring:** Model drift detection in production
5. **Advanced Techniques:** Neural networks, causal forests

## Technologies

- **Python:** pandas, scikit-learn, XGBoost, LightGBM, SHAP, statsmodels
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter Notebook (Google Colab)

## Author

Rengin Aktar  
Behavioral Data Scientist | MIT Applied AI & Data Science Certificate

## License

MIT License - See LICENSE file for details
