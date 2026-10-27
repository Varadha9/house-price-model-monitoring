"""
Main Entrypoint: House Price Prediction Production Monitoring System
Orchestrates:
1. Reference & Production Data Ingestion
2. Data Drift Analysis (KS-Test & PSI)
3. Concept Drift Analysis (RMSE, MAE, R², Residuals)
4. Visual Dashboard & HTML Report Generation
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_drift import DataDriftDetector
from src.concept_drift import ConceptDriftDetector
from src.visualizer import plot_monitoring_dashboard, generate_html_report

def main():
    print("=" * 78)
    print("  HOUSE PRICE PREDICTION - PRODUCTION MODEL MONITORING SYSTEM (ASSIGNMENT 5)")
    print("=" * 78)

    data_dir = os.path.join(BASE_DIR, 'data')
    models_dir = os.path.join(BASE_DIR, 'models')
    reports_dir = os.path.join(BASE_DIR, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    ref_path = os.path.join(data_dir, 'reference_data.csv')
    prod_path = os.path.join(data_dir, 'production_data.csv')
    model_path = os.path.join(models_dir, 'house_price_model.pkl')

    # Always ensure model and data exist
    if not (os.path.exists(ref_path) and os.path.exists(prod_path) and os.path.exists(model_path)):
        print("\n[!] Data or model artifacts missing. Running training pipeline first...")
        from models.train_model import train_and_save_model
        train_and_save_model()

    # Load reference & production data
    print(f"\n[1/4] Loading datasets...")
    ref_df = pd.read_csv(ref_path)
    prod_df = pd.read_csv(prod_path)
    print(f"  - Baseline Reference Data: {len(ref_df)} records")
    print(f"  - Production Batches:     {len(prod_df)} records across {prod_df['batch_id'].nunique()} time windows")

    # Load trained model
    print(f"\n[2/4] Loading production model artifact...")
    with open(model_path, 'rb') as f:
        model_payload = pickle.load(f)
    pipeline = model_payload['pipeline']
    baseline_metrics = model_payload['baseline_metrics']
    print(f"  - Baseline Holdout RMSE: ${baseline_metrics['rmse']:,.2f}")
    print(f"  - Baseline Holdout MAE:  ${baseline_metrics['mae']:,.2f}")
    print(f"  - Baseline Holdout R²:   {baseline_metrics['r2']:.4f}")

    # Compute baseline residuals on reference data
    features = ['area', 'bedrooms', 'bathrooms', 'age', 'location']
    ref_preds = pipeline.predict(ref_df[features])
    base_residuals = ref_df['price'].values - ref_preds

    # ------------------------------------------------------------------
    # Step 1: Data Drift Detection (Testing Month 2 feature shift)
    # ------------------------------------------------------------------
    print("\n[3/4] Running Data Drift Detection across Features...")
    drift_detector = DataDriftDetector(reference_df=ref_df, alpha=0.05, psi_threshold=0.20)

    # Evaluate each month for data drift
    batch_drift_summaries = {}
    for batch_id in sorted(prod_df['batch_id'].unique()):
        curr_batch_df = prod_df[prod_df['batch_id'] == batch_id]
        batch_name = f"Month {batch_id}"
        drift_res = drift_detector.detect_drift(curr_batch_df)
        batch_drift_summaries[batch_name] = drift_res

    # Display detailed feature breakdown for Month 2 (where data drift was introduced)
    m2_summary = batch_drift_summaries['Month 2']
    print("\n  >>> Feature-Level Data Drift Results (Month 2 vs Baseline):")
    print(f"  {'Feature':<12} {'Type':<12} {'Test':<15} {'Statistic':<12} {'p-value':<14} {'PSI':<8} {'Status'}")
    print("  " + "-" * 76)
    for feat, info in m2_summary['feature_drift'].items():
        if info['type'] == 'numeric':
            stat_str = f"KS={info['ks_statistic']:.3f}"
            p_str = f"{info['p_value']:.4e}"
        else:
            stat_str = f"Chi2={info['chi2_statistic']:.1f}"
            p_str = f"{info['p_value']:.4e}"
        psi_str = f"{info['psi']:.3f}"
        status_tag = f"[!] {info['status']}" if info['drift_detected'] else f"[OK] {info['status']}"
        print(f"  {feat:<12} {info['type']:<12} {info['test'][:14]:<15} {stat_str:<12} {p_str:<14} {psi_str:<8} {status_tag}")

    print(f"\n  Overall Dataset Drift Status: {m2_summary['overall_status']}")
    print(f"  Drifted Features: {m2_summary['drifted_features']}")

    # ------------------------------------------------------------------
    # Step 2: Concept Drift Detection (Tracking performance across batches)
    # ------------------------------------------------------------------
    print("\n[4/4] Running Concept Drift Detection Across Production Windows...")
    concept_detector = ConceptDriftDetector(
        model_pipeline=pipeline,
        baseline_metrics=baseline_metrics,
        baseline_residuals=base_residuals,
        rmse_threshold_pct=30.0,
        r2_floor=0.80
    )

    concept_results = []
    print(f"\n  {'Batch':<10} {'Scenario':<34} {'RMSE ($)':<12} {'MAE ($)':<12} {'R² Score':<10} {'RMSE Δ%':<10} {'Status'}")
    print("  " + "-" * 98)

    for batch_id in sorted(prod_df['batch_id'].unique()):
        curr_batch_df = prod_df[prod_df['batch_id'] == batch_id]
        batch_name = f"Month {batch_id}"
        scenario = curr_batch_df['scenario'].iloc[0]

        eval_res = concept_detector.evaluate_batch(curr_batch_df, batch_name=batch_name)
        eval_res['scenario'] = scenario
        concept_results.append(eval_res)

        m = eval_res['metrics']
        ind = eval_res['drift_indicators']
        status = eval_res['severity']
        delta_str = f"{ind['rmse_increase_pct']:>+7.1f}%"
        print(f"  {batch_name:<10} {scenario:<34} ${m['rmse']:<11,.0f} ${m['mae']:<11,.0f} {m['r2']:<10.4f} {delta_str:<10} {status}")

    # ------------------------------------------------------------------
    # Step 3: Visualization & Report Generation
    # ------------------------------------------------------------------
    dashboard_img = os.path.join(reports_dir, 'monitoring_dashboard.png')
    html_report = os.path.join(reports_dir, 'monitoring_report.html')
    json_metrics = os.path.join(reports_dir, 'monitoring_metrics.json')

    print("\nGenerating visual dashboard and HTML reports...")
    plot_monitoring_dashboard(ref_df, prod_df, concept_results, dashboard_img)
    generate_html_report(m2_summary, concept_results, dashboard_img, html_report)

    # Save programmatic metrics to JSON
    json_data = {
        'baseline_metrics': baseline_metrics,
        'data_drift_analysis': {
            b_name: {
                'drift_share': b_res['drift_share'],
                'dataset_drift': b_res['dataset_drift'],
                'drifted_features': b_res['drifted_features']
            }
            for b_name, b_res in batch_drift_summaries.items()
        },
        'concept_drift_analysis': [
            {
                'batch': r['batch_name'],
                'scenario': r['scenario'],
                'metrics': r['metrics'],
                'indicators': r['drift_indicators'],
                'severity': r['severity']
            }
            for r in concept_results
        ]
    }
    with open(json_metrics, 'w') as jf:
        json.dump(json_data, jf, indent=2)
    print(f"Machine-readable metrics saved to: {json_metrics}")

    print("\n" + "=" * 78)
    print("  MONITORING PIPELINE RUN COMPLETE!")
    print(f"  - Dashboard Visuals: {dashboard_img}")
    print(f"  - Interactive Report: {html_report}")
    print(f"  - Metrics Summary:    {json_metrics}")
    print("=" * 78)

if __name__ == '__main__':
    main()
