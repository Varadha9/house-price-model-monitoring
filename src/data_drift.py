"""
Data Drift Detection Module
Detects covariate / input feature drift using:
- Kolmogorov-Smirnov (KS) 2-Sample Test (Continuous features: area, bedrooms, bathrooms, age)
- Population Stability Index (PSI) (Continuous & categorical features)
- Chi-Square Goodness-of-Fit Test (Categorical features: location)
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare

def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Calculate the Population Stability Index (PSI) between reference and current feature distributions.
    Interpretation:
      - PSI < 0.10: No significant shift / Stable
      - 0.10 <= PSI < 0.25: Moderate shift / Slight drift
      - PSI >= 0.25: Significant shift / Severe drift
    """
    # Define quantile bins based on reference data
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(reference, quantiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    # Handle duplicate bin edges if values are discrete
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)

    ref_pct = (ref_counts / len(reference)) + epsilon
    curr_pct = (curr_counts / len(current)) + epsilon

    # Normalize back to sum to 1
    ref_pct = ref_pct / np.sum(ref_pct)
    curr_pct = curr_pct / np.sum(curr_pct)

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(np.round(psi_val, 4))

class DataDriftDetector:
    def __init__(self, reference_df: pd.DataFrame, alpha: float = 0.05, psi_threshold: float = 0.20):
        self.reference_df = reference_df
        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.numeric_features = ['area', 'bedrooms', 'bathrooms', 'age']
        self.categorical_features = ['location']

    def detect_numeric_drift(self, current_df: pd.DataFrame) -> dict:
        results = {}
        for feature in self.numeric_features:
            ref_vals = self.reference_df[feature].values
            curr_vals = current_df[feature].values

            # Two-sample Kolmogorov-Smirnov test
            ks_res = ks_2samp(ref_vals, curr_vals)
            statistic = float(np.round(ks_res.statistic, 4))
            p_val = float(ks_res.pvalue)

            # PSI
            psi = calculate_psi(ref_vals, curr_vals)

            # Decision: statistical significance (p < alpha) and effect size
            drift_detected = bool(p_val < self.alpha and (statistic > 0.08 or psi >= 0.10))

            results[feature] = {
                'type': 'numeric',
                'test': 'Two-Sample Kolmogorov-Smirnov (KS)',
                'ks_statistic': statistic,
                'p_value': p_val,
                'psi': psi,
                'ref_mean': float(np.round(np.mean(ref_vals), 2)),
                'curr_mean': float(np.round(np.mean(curr_vals), 2)),
                'ref_std': float(np.round(np.std(ref_vals), 2)),
                'curr_std': float(np.round(np.std(curr_vals), 2)),
                'drift_detected': drift_detected,
                'status': 'DRIFT DETECTED' if drift_detected else 'STABLE'
            }
        return results

    def detect_categorical_drift(self, current_df: pd.DataFrame) -> dict:
        results = {}
        for feature in self.categorical_features:
            ref_counts = self.reference_df[feature].value_counts(normalize=True)
            categories = ref_counts.index.tolist()

            # Align current counts to categories
            curr_series = current_df[feature].value_counts()
            observed = np.array([curr_series.get(cat, 0) for cat in categories], dtype=float)
            total_curr = len(current_df)

            # Expected frequencies based on reference proportions
            expected = np.array([ref_counts[cat] * total_curr for cat in categories], dtype=float)

            # Small count correction
            observed = np.maximum(observed, 1e-4)
            expected = np.maximum(expected, 1e-4)

            # Rescale expected to exactly equal sum of observed
            expected = expected * (np.sum(observed) / np.sum(expected))

            chi2_stat, p_val = chisquare(f_obs=observed, f_exp=expected)
            chi2_stat = float(np.round(chi2_stat, 4))
            p_val = float(p_val)

            # Categorical PSI
            ref_pct = np.array([ref_counts[cat] for cat in categories])
            curr_pct = observed / np.sum(observed)
            psi_val = float(np.round(np.sum((curr_pct - ref_pct) * np.log((curr_pct + 1e-5) / (ref_pct + 1e-5))), 4))

            drift_detected = bool(p_val < self.alpha and psi_val >= 0.10)

            ref_dist = {cat: float(np.round(ref_counts[cat], 3)) for cat in categories}
            curr_dist = {cat: float(np.round(curr_series.get(cat, 0) / total_curr, 3)) for cat in categories}

            results[feature] = {
                'type': 'categorical',
                'test': 'Chi-Square Goodness-of-Fit',
                'chi2_statistic': chi2_stat,
                'p_value': p_val,
                'psi': psi_val,
                'ref_distribution': ref_dist,
                'curr_distribution': curr_dist,
                'drift_detected': drift_detected,
                'status': 'DRIFT DETECTED' if drift_detected else 'STABLE'
            }
        return results

    def detect_drift(self, current_df: pd.DataFrame) -> dict:
        """Run full data drift analysis over all features."""
        num_res = self.detect_numeric_drift(current_df)
        cat_res = self.detect_categorical_drift(current_df)
        all_features = {**num_res, **cat_res}

        drifted_features = [f for f, res in all_features.items() if res['drift_detected']]
        total_features = len(all_features)
        dataset_drift = (len(drifted_features) / total_features) >= 0.40

        return {
            'feature_drift': all_features,
            'drifted_features': drifted_features,
            'drift_share': float(np.round(len(drifted_features) / total_features, 3)),
            'dataset_drift': dataset_drift,
            'overall_status': 'DATA DRIFT DETECTED' if dataset_drift else 'DATA STABLE'
        }
