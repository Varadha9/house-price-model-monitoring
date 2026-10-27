"""
Concept Drift Detection Module
Detects shifts in the relationship between input features and target (P(y|X) drift):
- Performance Degradation Tracking (RMSE, MAE, R², MAPE) vs. Baseline
- Residual Error Distribution Shift (KS-Test on prediction errors: e = y - y_hat)
- Systemic Prediction Bias & Severity Alerting
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ConceptDriftDetector:
    def __init__(self, model_pipeline, baseline_metrics: dict, baseline_residuals: np.ndarray,
                 rmse_threshold_pct: float = 30.0, r2_floor: float = 0.80, alpha: float = 0.05):
        """
        :param model_pipeline: Trained sklearn Pipeline
        :param baseline_metrics: {'rmse': float, 'mae': float, 'r2': float}
        :param baseline_residuals: Array of y_true - y_pred from baseline holdout set
        :param rmse_threshold_pct: Percentage increase in RMSE to trigger alert (e.g. +30%)
        :param r2_floor: Minimum acceptable R2 score before flagging drift
        :param alpha: Significance level for residual distribution test
        """
        self.model = model_pipeline
        self.baseline_metrics = baseline_metrics
        self.baseline_residuals = baseline_residuals
        self.rmse_threshold_pct = rmse_threshold_pct
        self.r2_floor = r2_floor
        self.alpha = alpha

    def evaluate_batch(self, current_df: pd.DataFrame, batch_name: str = "Current Batch") -> dict:
        features = ['area', 'bedrooms', 'bathrooms', 'age', 'location']
        X_curr = current_df[features]
        y_curr = current_df['price'].values

        # Model predictions
        y_pred = self.model.predict(X_curr)
        residuals = y_curr - y_pred

        # Metrics
        curr_rmse = float(np.sqrt(mean_squared_error(y_curr, y_pred)))
        curr_mae = float(mean_absolute_error(y_curr, y_pred))
        curr_r2 = float(r2_score(y_curr, y_pred))
        curr_mape = float(np.mean(np.abs(residuals / np.maximum(y_curr, 1e-4))) * 100)

        # Baseline comparisons
        base_rmse = self.baseline_metrics['rmse']
        base_r2 = self.baseline_metrics['r2']
        base_mae = self.baseline_metrics['mae']

        rmse_increase_pct = float(np.round(((curr_rmse - base_rmse) / base_rmse) * 100, 2))
        r2_drop = float(np.round(base_r2 - curr_r2, 4))
        mean_residual_bias = float(np.round(np.mean(residuals), 2))

        # Residual distribution shift test (KS test on errors)
        ks_res = ks_2samp(self.baseline_residuals, residuals)
        res_ks_stat = float(np.round(ks_res.statistic, 4))
        res_p_val = float(ks_res.pvalue)
        residual_drift_detected = bool(res_p_val < self.alpha and res_ks_stat > 0.15)

        # Performance drift triggers
        rmse_drift = rmse_increase_pct >= self.rmse_threshold_pct
        r2_drift = curr_r2 < self.r2_floor

        concept_drift_detected = bool(rmse_drift or r2_drift or (residual_drift_detected and rmse_increase_pct > 20.0))

        if rmse_increase_pct >= 80.0 or curr_r2 < 0.60:
            severity = "CRITICAL CONCEPT DRIFT"
            status = "CRITICAL"
        elif concept_drift_detected:
            severity = "MODERATE CONCEPT DRIFT"
            status = "WARNING"
        else:
            severity = "NO CONCEPT DRIFT"
            status = "STABLE"

        return {
            'batch_name': batch_name,
            'metrics': {
                'rmse': float(np.round(curr_rmse, 2)),
                'mae': float(np.round(curr_mae, 2)),
                'r2': float(np.round(curr_r2, 4)),
                'mape': float(np.round(curr_mape, 2))
            },
            'baseline_metrics': {
                'rmse': float(np.round(base_rmse, 2)),
                'mae': float(np.round(base_mae, 2)),
                'r2': float(np.round(base_r2, 4))
            },
            'drift_indicators': {
                'rmse_increase_pct': rmse_increase_pct,
                'r2_drop': r2_drop,
                'mean_residual_bias': mean_residual_bias,
                'residual_ks_stat': res_ks_stat,
                'residual_ks_p_val': res_p_val,
                'residual_drift_detected': residual_drift_detected
            },
            'concept_drift_detected': concept_drift_detected,
            'severity': severity,
            'status': status,
            'predictions': y_pred,
            'residuals': residuals
        }
