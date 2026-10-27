# MLOps Assignment 5 — Production Model Monitoring System

**Project**: House Price Prediction Model Monitoring (Data Drift, Concept Drift, and Visual Reporting)  
**Author**: Varad  
**Environment**: Python 3.13 / Scikit-learn / SciPy / Pandas / Matplotlib  

---

## 1. Problem Statement

A **House Price Prediction Model** has been deployed in production to forecast property valuations based on home features such as square footage (`area`), number of bedrooms (`bedrooms`), number of bathrooms (`bathrooms`), geographic sector (`location`), and property age (`age`).

Over time in production environments, machine learning models experience performance decay due to:
1. **Data Drift (Covariate Shift)**: Changes in the distribution of input features $P(X)$ over time (e.g., changes in user demographics, urban expansion towards luxury or suburban homes).
2. **Concept Drift**: Changes in the true mathematical relationship between features and the target variable $P(y|X)$ (e.g., macro-economic inflation, interest rate hikes, or structural real estate market revaluations).

The objective of Assignment 5 is to build an automated, lightweight, production-grade monitoring system that:
- Detects **Data Drift** across both continuous and categorical input features.
- Detects **Concept Drift** through statistical tracking of model performance metrics ($RMSE, MAE, R^2, MAPE$) and prediction residual distributions.
- Generates an **Executive Visual Dashboard (PNG)** and a portable, self-contained **Interactive HTML Monitoring Report**.

---

## 2. System Architecture

```text
MLOps A5/
└── Assignments-5/
    ├── data/
    │   ├── reference_data.csv        # Baseline training/holdout dataset (1,500 samples)
    │   └── production_data.csv       # 4 sequential production batches / months (1,600 samples)
    ├── models/
    │   ├── train_model.py            # Gradient Boosting pipeline trainer & serializer
    │   └── house_price_model.pkl     # Serialized Scikit-learn Pipeline & baseline metrics
    ├── src/
    │   ├── data_drift.py             # KS-Test, PSI, and Chi-Square detection algorithms
    │   ├── concept_drift.py          # Metric degradation tracking and residual bias analysis
    │   └── visualizer.py             # Matplotlib multi-panel dashboard & HTML generator
    ├── reports/
    │   ├── monitoring_dashboard.png  # 6-panel high-resolution monitoring dashboard
    │   ├── monitoring_report.html    # Standalone interactive executive report (embedded base64)
    │   └── monitoring_metrics.json   # Machine-readable metric payload
    ├── run_monitoring.py             # End-to-end execution runner
    ├── requirements.txt              # Project dependencies
    ├── README.md                     # Quickstart documentation
    └── document.md                   # Complete assignment submission report
```

---

## 3. Methodology & Mathematical Formulation

### 3.1 Data Drift Detection
1. **Two-Sample Kolmogorov-Smirnov (KS) Test** (for continuous features: `area`, `bedrooms`, `bathrooms`, `age`):
   Compares the empirical cumulative distribution functions $F_{ref}(x)$ and $F_{curr}(x)$:
   $$D = \sup_x |F_{ref}(x) - F_{curr}(x)|$$
   If the $p$-value is below the significance threshold $\alpha = 0.05$ and effect size $D > 0.08$, statistical data drift is flagged.

2. **Population Stability Index (PSI)**:
   Quantifies distribution divergence across quantile bins:
   $$\text{PSI} = \sum_{i=1}^{B} (P_{curr, i} - P_{ref, i}) \times \ln\left(\frac{P_{curr, i}}{P_{ref, i}}\right)$$
   - $\text{PSI} < 0.10$: Stable / No significant drift
   - $0.10 \le \text{PSI} < 0.25$: Moderate drift (warning)
   - $\text{PSI} \ge 0.25$: Severe drift (action required)

3. **Chi-Square Goodness-of-Fit Test** (for categorical feature: `location`):
   Tests whether current categorical frequency counts deviate from baseline expected frequencies:
   $$\chi^2 = \sum_{k} \frac{(O_k - E_k)^2}{E_k}$$

### 3.2 Concept Drift Detection
1. **Performance Metrics Tracking vs. Baseline**:
   Monitors percentage degradation in Root Mean Squared Error ($RMSE$) and $R^2$ score:
   $$\Delta \text{RMSE} = \frac{\text{RMSE}_{curr} - \text{RMSE}_{base}}{\text{RMSE}_{base}} \times 100\%$$
   $$\Delta R^2 = R^2_{base} - R^2_{curr}$$
   - Warning trigger: $\Delta \text{RMSE} \ge +30\%$ or $R^2 < 0.80$
   - Critical trigger: $\Delta \text{RMSE} \ge +80\%$ or $R^2 < 0.60$

2. **Residual Error Distribution Test**:
   Tests whether prediction residuals $e = y - \hat{y}$ deviate from baseline error residuals using two-sample KS-testing to uncover systemic under- or over-estimation.

---

## 4. Production Simulation Scenarios

Production traffic was simulated across four sequential monthly windows ($N = 400$ per window):
- **Month 1 (Normal Production)**: Standard in-distribution inputs following reference statistics.
- **Month 2 (Data Drift)**: Buyers demand larger houses (mean square footage shifts from 1,850 sq ft to 2,600 sq ft) and geographic transactions shift towards Downtown (55%) and Suburban (25%). The underlying pricing formula is unchanged.
- **Month 3 (Concept Drift)**: Macroeconomic inflation increases cost per square foot by $+45\%$ and inflates location premiums. Feature distributions return to normal, but model performance drops dramatically.
- **Month 4 (Compound Drift)**: Simultaneous severe data drift and concept drift.

---

## 5. Execution and Terminal Verification

### 5.1 Running the Monitoring Pipeline
```bash
python3 run_monitoring.py
```

### 5.2 Terminal Output
```text
==============================================================================
  HOUSE PRICE PREDICTION - PRODUCTION MODEL MONITORING SYSTEM (ASSIGNMENT 5)
==============================================================================

[1/4] Loading datasets...
  - Baseline Reference Data: 1500 records
  - Production Batches:     1600 records across 4 time windows

[2/4] Loading production model artifact...
  - Baseline Holdout RMSE: $18,973.99
  - Baseline Holdout MAE:  $15,156.01
  - Baseline Holdout R²:   0.9641

[3/4] Running Data Drift Detection across Features...

  >>> Feature-Level Data Drift Results (Month 2 vs Baseline):
  Feature      Type         Test            Statistic    p-value        PSI      Status
  ----------------------------------------------------------------------------
  area         numeric      Two-Sample Kol  KS=0.596     1.5317e-105    2.093    [!] DRIFT DETECTED
  bedrooms     numeric      Two-Sample Kol  KS=0.231     3.0180e-15     0.391    [!] DRIFT DETECTED
  bathrooms    numeric      Two-Sample Kol  KS=0.388     5.2872e-43     0.634    [!] DRIFT DETECTED
  age          numeric      Two-Sample Kol  KS=0.147     1.9812e-06     0.783    [!] DRIFT DETECTED
  location     categorical  Chi-Square Goo  Chi2=508.5   6.7107e-110    0.785    [!] DRIFT DETECTED

  Overall Dataset Drift Status: DATA DRIFT DETECTED
  Drifted Features: ['area', 'bedrooms', 'bathrooms', 'age', 'location']

[4/4] Running Concept Drift Detection Across Production Windows...

  Batch      Scenario                           RMSE ($)     MAE ($)      R² Score   RMSE Δ%    Status
  --------------------------------------------------------------------------------------------------
  Month 1    Normal (No Drift)                  $17,875      $14,377      0.9648        -5.8%   NO CONCEPT DRIFT
  Month 2    Data Drift (Feature Shift)         $29,188      $21,497      0.9192       +53.8%   MODERATE CONCEPT DRIFT
  Month 3    Concept Drift (Relationship Shift) $256,097     $251,969     -2.9574    +1249.7%   CRITICAL CONCEPT DRIFT
  Month 4    Severe Compound Drift (Data + Concept) $473,470     $466,587     -6.3822    +2395.4%   CRITICAL CONCEPT DRIFT

Generating visual dashboard and HTML reports...
Monitoring Dashboard saved to: reports/monitoring_dashboard.png
HTML Monitoring Report saved to: reports/monitoring_report.html
Machine-readable metrics saved to: reports/monitoring_metrics.json

==============================================================================
  MONITORING PIPELINE RUN COMPLETE!
==============================================================================
```

---

## 6. Generated Visualizations & Reports

### 6.1 Monitoring Dashboard (`reports/monitoring_dashboard.png`)
The generated 6-panel monitoring dashboard provides immediate visual clarity:
1. **Area Distribution Shift**: Baseline histogram (blue) vs Month 2 distribution (red), revealing a pronounced right-shift into luxury home areas.
2. **Bedrooms Frequency Comparison**: Reference vs Month 2 grouped bars showing elevated shares of 4-bedroom and 5-bedroom properties.
3. **Location Frequency Breakdown**: Highlights Downtown transaction surge from 15% to 55%.
4. **Error Metrics over Time**: Production RMSE and MAE against baseline threshold lines.
5. **R² Score Degradation**: R² drops from 0.964 to negative values during concept drift.
6. **Residual Error Shift Violin Plot**: Residuals centered at $0 in Month 1 shift sharply positive in Month 3 and 4, confirming severe systematic under-prediction.

### 6.2 Standalone HTML Executive Report (`reports/monitoring_report.html`)
- Built with responsive CSS card components and status pills.
- Includes embedded Base64 visualization to remain 100% portable without broken local asset paths.
- Features detailed scorecard tables for both feature drift and performance degradation.
- Delivers actionable automated remediation recommendations: trigger automated retraining pipeline `v2.0.0` using recent window data.

---

## 7. Key Findings & Conclusion

1. **Decoupled Drift Types**: Data drift can occur without catastrophic model failure (Month 2 maintained an $R^2$ of $0.9192$ despite input feature drift), whereas concept drift (Month 3) immediately degrades model accuracy even if input feature distributions look identical.
2. **Actionable Alerts**: By combining KS-test, PSI, and residual tracking, the monitoring system distinguishes between input changes and pricing model invalidation, enabling targeted CI/CD model retraining.
