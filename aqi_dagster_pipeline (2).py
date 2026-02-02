# ============================================================
# AQI PREDICTION DAGSTER PIPELINE - FOR LOCAL EXECUTION
# Run with: dagster dev -f aqi_dagster_pipeline.py
# ============================================================

from dagster import asset, Definitions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

# UPDATE THIS PATH TO YOUR CSV FILE LOCATION
DATA_PATH = 'Air_quality_data.csv'

@asset
def raw_data():
    """Load AQI dataset"""
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

@asset
def cleaned_data(raw_data):
    """Clean and preprocess the data"""
    df = raw_data.copy()
    df = df.dropna(subset=['AQI'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek

    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    for col in pollutant_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    df = df.dropna(thresh=len(df.columns) - 3)
    print(f"✅ Data cleaned: {df.shape[0]} rows")
    return df

@asset
def eda_results(cleaned_data):
    """Perform EDA"""
    results = {
        'total_records': len(cleaned_data),
        'cities': cleaned_data['City'].nunique(),
        'aqi_mean': cleaned_data['AQI'].mean()
    }
    print(f"✅ EDA completed")
    return results

@asset
def preprocessed_data(cleaned_data):
    """Prepare data for ML"""
    feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO',
                    'SO2', 'O3', 'Year', 'Month', 'Day', 'DayOfWeek']

    df_model = cleaned_data[feature_cols + ['AQI']].dropna()
    X = df_model[feature_cols]
    y = df_model['AQI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✅ Preprocessing complete: {len(X_train)} train, {len(X_test)} test")
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': feature_cols
    }

@asset
def linear_regression_model(preprocessed_data):
    """Train Linear Regression"""
    data = preprocessed_data
    model = LinearRegression()
    model.fit(data['X_train'], data['y_train'])
    y_pred = model.predict(data['X_test'])

    metrics = {
        'model_name': 'Linear Regression',
        'test_r2': r2_score(data['y_test'], y_pred),
        'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
        'mae': mean_absolute_error(data['y_test'], y_pred)
    }
    print(f"✅ Linear Regression: R²={metrics['test_r2']:.4f}")
    return {'model': model, 'metrics': metrics}

@asset
def decision_tree_model(preprocessed_data):
    """Train Decision Tree"""
    data = preprocessed_data
    model = DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42)
    model.fit(data['X_train'], data['y_train'])
    y_pred = model.predict(data['X_test'])

    metrics = {
        'model_name': 'Decision Tree',
        'test_r2': r2_score(data['y_test'], y_pred),
        'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
        'mae': mean_absolute_error(data['y_test'], y_pred)
    }
    print(f"✅ Decision Tree: R²={metrics['test_r2']:.4f}")
    return {'model': model, 'metrics': metrics}

@asset
def random_forest_model(preprocessed_data):
    """Train Random Forest"""
    data = preprocessed_data
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(data['X_train'], data['y_train'])
    y_pred = model.predict(data['X_test'])

    metrics = {
        'model_name': 'Random Forest',
        'test_r2': r2_score(data['y_test'], y_pred),
        'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
        'mae': mean_absolute_error(data['y_test'], y_pred)
    }
    print(f"✅ Random Forest: R²={metrics['test_r2']:.4f}")
    return {'model': model, 'metrics': metrics}

@asset
def gradient_boosting_model(preprocessed_data):
    """Train Gradient Boosting"""
    data = preprocessed_data
    model = GradientBoostingRegressor(n_estimators=100, max_depth=7, random_state=42)
    model.fit(data['X_train'], data['y_train'])
    y_pred = model.predict(data['X_test'])

    metrics = {
        'model_name': 'Gradient Boosting',
        'test_r2': r2_score(data['y_test'], y_pred),
        'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
        'mae': mean_absolute_error(data['y_test'], y_pred)
    }
    print(f"✅ Gradient Boosting: R²={metrics['test_r2']:.4f}")
    return {'model': model, 'metrics': metrics}

@asset
def model_comparison(linear_regression_model, decision_tree_model,
                     random_forest_model, gradient_boosting_model):
    """Compare all models"""
    models = [linear_regression_model, decision_tree_model,
              random_forest_model, gradient_boosting_model]
    comparison_df = pd.DataFrame([m['metrics'] for m in models])

    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(comparison_df.to_string(index=False))

    best_idx = comparison_df['test_r2'].idxmax()
    print(f"\n🏆 Best Model: {comparison_df.loc[best_idx, 'model_name']}")
    return comparison_df

# Define all assets
defs = Definitions(
    assets=[
        raw_data,
        cleaned_data,
        eda_results,
        preprocessed_data,
        linear_regression_model,
        decision_tree_model,
        random_forest_model,
        gradient_boosting_model,
        model_comparison
    ]
)
