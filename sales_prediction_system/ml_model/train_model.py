"""
Sales Prediction Model Training Script
=======================================
Project: Sales Prediction System
Team: Susmita Biswas, Jotirmoy Mondal, Alinda Paul, Suchi Nondi

Run this script ONCE before starting the Django server.
It trains the model on the Advertising dataset and saves model.pkl.

Usage:
    python ml_model/train_model.py
"""

import os
import sys

# Add parent dir to path so we can run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT = True
except ImportError:
    PLOT = False

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=" * 55)
print("  Sales Prediction Model — Training Script")
print("=" * 55)

# Step 1: Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
print(f"\n[1] Loading dataset from:\n    {url}")

try:
    df = pd.read_csv(url)
    print(f"    ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"    ✗ Could not load from URL: {e}")
    print("    Using synthetic fallback data for demo...")
    # Synthetic data with similar distribution
    np.random.seed(42)
    n = 200
    tv = np.random.uniform(0.7, 296.4, n)
    radio = np.random.uniform(0.0, 49.6, n)
    newspaper = np.random.uniform(0.3, 114.0, n)
    sales = 0.0458 * tv + 0.1885 * radio + 0.00115 * newspaper + 2.9389 + np.random.normal(0, 1.5, n)
    df = pd.DataFrame({'TV': tv, 'radio': radio, 'newspaper': newspaper, 'sales': sales})

# Step 2: Explore
print("\n[2] Dataset overview:")
print(df.head(3).to_string())
print(f"\n    Shape: {df.shape}")
print(f"    Missing values: {df.isnull().sum().sum()}")

# Step 3: Features & target
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[3] Train: {len(X_train)}, Test: {len(X_test)}")

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n[4] Model trained.")
print(f"    Coefficients: TV={model.coef_[0]:.4f}, Radio={model.coef_[1]:.4f}, Newspaper={model.coef_[2]:.4f}")
print(f"    Intercept: {model.intercept_:.4f}")

# Step 6: Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\n[5] Evaluation:")
print(f"    MAE : {mae:.4f}")
print(f"    MSE : {mse:.4f}")
print(f"    RMSE: {mse**0.5:.4f}")

# Step 7: Save model
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\n[6] Model saved → {model_path}")

# Step 8: Optional plots
if PLOT:
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='royalblue', edgecolors='white', linewidth=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.tight_layout()
    plot_path = os.path.join(model_dir, 'prediction_plot.png')
    plt.savefig(plot_path, dpi=120)
    print(f"    Plot saved → {plot_path}")

print("\n✓ Done! You can now start the Django server.\n")
