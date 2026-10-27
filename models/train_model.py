import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

def generate_reference_data(n_samples=1500):
    """
    Generate baseline reference dataset representing historical / training & validation data.
    """
    area = np.random.normal(loc=1850, scale=420, size=n_samples).clip(750, 4200)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.10, 0.25, 0.40, 0.20, 0.05])
    bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.4, n_samples)), 1, 4).astype(int)
    locations = np.random.choice(['Suburban', 'Urban', 'Downtown', 'Rural'], size=n_samples, p=[0.40, 0.30, 0.15, 0.15])
    age = np.random.uniform(1, 35, size=n_samples)
    
    loc_effect = {'Downtown': 120000, 'Urban': 70000, 'Suburban': 40000, 'Rural': 5000}
    loc_val = np.array([loc_effect[loc] for loc in locations])
    
    # Ground truth baseline relationship
    price = (
        60000 +
        (area * 175) +
        (bedrooms * 22000) +
        (bathrooms * 28000) +
        loc_val -
        (age * 1100) +
        np.random.normal(0, 15000, size=n_samples)
    )
    
    df = pd.DataFrame({
        'area': np.round(area, 1),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'location': locations,
        'age': np.round(age, 1),
        'price': np.round(price, 2)
    })
    return df

def generate_production_batches(n_per_batch=400):
    """
    Simulate production batches across 4 temporal windows (months):
    - Batch 1 (Month 1): In-distribution (No drift)
    - Batch 2 (Month 2): Data Drift in input features (Area shift & Location distribution shift)
    - Batch 3 (Month 3): Concept Drift (Real estate market inflation / structural pricing shift)
    - Batch 4 (Month 4): Combined Data Drift + Concept Drift
    """
    batches = []
    loc_effect_base = {'Downtown': 120000, 'Urban': 70000, 'Suburban': 40000, 'Rural': 5000}
    
    for batch_id in [1, 2, 3, 4]:
        if batch_id == 1:
            # Normal in-distribution
            area = np.random.normal(loc=1850, scale=420, size=n_per_batch).clip(750, 4200)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_per_batch, p=[0.10, 0.25, 0.40, 0.20, 0.05])
            bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.4, n_per_batch)), 1, 4).astype(int)
            locations = np.random.choice(['Suburban', 'Urban', 'Downtown', 'Rural'], size=n_per_batch, p=[0.40, 0.30, 0.15, 0.15])
            age = np.random.uniform(1, 35, size=n_per_batch)
            loc_val = np.array([loc_effect_base[loc] for loc in locations])
            price = (
                60000 + (area * 175) + (bedrooms * 22000) + (bathrooms * 28000) +
                loc_val - (age * 1100) + np.random.normal(0, 15000, size=n_per_batch)
            )
            scenario = "Normal (No Drift)"
            
        elif batch_id == 2:
            # Data Drift: Area significantly larger (luxury houses), Location heavily Downtown/Suburban
            area = np.random.normal(loc=2600, scale=500, size=n_per_batch).clip(1000, 4800)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_per_batch, p=[0.02, 0.12, 0.36, 0.36, 0.14])
            bathrooms = np.clip(np.round(bedrooms * 0.85 + np.random.normal(0, 0.3, n_per_batch)), 1, 5).astype(int)
            locations = np.random.choice(['Suburban', 'Urban', 'Downtown', 'Rural'], size=n_per_batch, p=[0.25, 0.15, 0.55, 0.05])
            age = np.random.uniform(1, 30, size=n_per_batch)
            loc_val = np.array([loc_effect_base[loc] for loc in locations])
            # Price relationship follows the true underlying formula
            price = (
                60000 + (area * 175) + (bedrooms * 22000) + (bathrooms * 28000) +
                loc_val - (age * 1100) + np.random.normal(0, 15000, size=n_per_batch)
            )
            scenario = "Data Drift (Feature Shift)"
            
        elif batch_id == 3:
            # Concept Drift: Inputs are standard baseline, but market dynamics shifted (macro inflation + price per sqft surge)
            area = np.random.normal(loc=1850, scale=420, size=n_per_batch).clip(750, 4200)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_per_batch, p=[0.10, 0.25, 0.40, 0.20, 0.05])
            bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.4, n_per_batch)), 1, 4).astype(int)
            locations = np.random.choice(['Suburban', 'Urban', 'Downtown', 'Rural'], size=n_per_batch, p=[0.40, 0.30, 0.15, 0.15])
            age = np.random.uniform(1, 35, size=n_per_batch)
            
            # Underlying market inflation & structural price increase (+45% per sqft, downtown premium increased)
            loc_effect_drift = {'Downtown': 190000, 'Urban': 105000, 'Suburban': 60000, 'Rural': 10000}
            loc_val_drift = np.array([loc_effect_drift[loc] for loc in locations])
            price = (
                95000 + (area * 250) + (bedrooms * 30000) + (bathrooms * 36000) +
                loc_val_drift - (age * 800) + np.random.normal(0, 18000, size=n_per_batch)
            )
            scenario = "Concept Drift (Relationship Shift)"
            
        else: # batch_id == 4
            # Severe Combined Drift: Inputs shifted AND market relationship shifted
            area = np.random.normal(loc=2700, scale=550, size=n_per_batch).clip(1100, 5000)
            bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_per_batch, p=[0.02, 0.10, 0.32, 0.38, 0.18])
            bathrooms = np.clip(np.round(bedrooms * 0.85 + np.random.normal(0, 0.3, n_per_batch)), 1, 5).astype(int)
            locations = np.random.choice(['Suburban', 'Urban', 'Downtown', 'Rural'], size=n_per_batch, p=[0.20, 0.10, 0.65, 0.05])
            age = np.random.uniform(1, 22, size=n_per_batch)
            
            loc_effect_drift = {'Downtown': 210000, 'Urban': 110000, 'Suburban': 65000, 'Rural': 12000}
            loc_val_drift = np.array([loc_effect_drift[loc] for loc in locations])
            price = (
                110000 + (area * 270) + (bedrooms * 32000) + (bathrooms * 40000) +
                loc_val_drift - (age * 600) + np.random.normal(0, 20000, size=n_per_batch)
            )
            scenario = "Severe Compound Drift (Data + Concept)"
        
        batch_df = pd.DataFrame({
            'batch_id': batch_id,
            'time_window': f'Month {batch_id}',
            'scenario': scenario,
            'area': np.round(area, 1),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'location': locations,
            'age': np.round(age, 1),
            'price': np.round(price, 2)
        })
        batches.append(batch_df)
        
    return pd.concat(batches, ignore_index=True)

def train_and_save_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Generating reference dataset...")
    df_ref = generate_reference_data(n_samples=1500)
    ref_path = os.path.join(data_dir, 'reference_data.csv')
    df_ref.to_csv(ref_path, index=False)
    print(f"Saved reference data ({len(df_ref)} rows) to: {ref_path}")

    print("Generating production stream data (4 batches)...")
    df_prod = generate_production_batches(n_per_batch=400)
    prod_path = os.path.join(data_dir, 'production_data.csv')
    df_prod.to_csv(prod_path, index=False)
    print(f"Saved production data ({len(df_prod)} rows) to: {prod_path}")

    # Model training with train/val holdout
    numeric_features = ['area', 'bedrooms', 'bathrooms', 'age']
    categorical_features = ['location']
    all_features = numeric_features + categorical_features

    train_df, val_df = train_test_split(df_ref, test_size=0.25, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42))
    ])

    X_train = train_df[all_features]
    y_train = train_df['price']
    X_val = val_df[all_features]
    y_val = val_df['price']

    print("Training GradientBoostingRegressor pipeline...")
    model_pipeline.fit(X_train, y_train)

    # Validation holdout baseline metrics (out-of-sample)
    y_val_pred = model_pipeline.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))

    print(f"Baseline Validation Performance (Holdout Set):")
    print(f"  RMSE: ${val_rmse:,.2f}")
    print(f"  MAE:  ${val_mae:,.2f}")
    print(f"  R²:   {val_r2:.4f}")

    # Serialize model and metadata
    model_path = os.path.join(models_dir, 'house_price_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'pipeline': model_pipeline,
            'features': all_features,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'baseline_metrics': {'rmse': val_rmse, 'mae': val_mae, 'r2': val_r2}
        }, f)
    print(f"Model successfully serialized to: {model_path}")

if __name__ == '__main__':
    train_and_save_model()
