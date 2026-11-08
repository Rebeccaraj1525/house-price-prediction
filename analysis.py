import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Load Data ─────────────────────────────────────
df = pd.read_csv("Housing.csv")

# ── Basic Info ────────────────────────────────────
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Stats:")
print(df.describe())

# ── Chart 1: Price Distribution ───────────────────
plt.figure(figsize=(10, 5))
df["price"].hist(bins=30, color="#2196F3", edgecolor="white")
plt.title("Distribution of House Prices", fontsize=16, fontweight="bold")
plt.xlabel("Price", fontsize=13)
plt.ylabel("Number of Houses", fontsize=13)
plt.tight_layout()
plt.savefig("chart1_price_distribution.png", dpi=150)
plt.show()
print("Chart 1 done ✅")

# ── Chart 2: Price vs Area ─────────────────────────
plt.figure(figsize=(10, 6))
plt.scatter(df["area"], df["price"], alpha=0.5,
            color="#1F3864", edgecolors="white", linewidth=0.5)
plt.title("House Price vs Area", fontsize=16, fontweight="bold")
plt.xlabel("Area (sq ft)", fontsize=13)
plt.ylabel("Price", fontsize=13)
plt.tight_layout()
plt.savefig("chart2_price_vs_area.png", dpi=150)
plt.show()
print("Chart 2 done ✅")

# ── Chart 3: Average Price by Number of Bedrooms ──
plt.figure(figsize=(8, 5))
avg_price_bed = df.groupby("bedrooms")["price"].mean().reset_index()
sns.barplot(x="bedrooms", y="price", data=avg_price_bed,
            palette="Blues_d")
plt.title("Average Price by Number of Bedrooms",
          fontsize=16, fontweight="bold")
plt.xlabel("Number of Bedrooms", fontsize=13)
plt.ylabel("Average Price", fontsize=13)
plt.tight_layout()
plt.savefig("chart3_price_by_bedrooms.png", dpi=150)
plt.show()
print("Chart 3 done ✅")

# ── Chart 4: Correlation Heatmap ──────────────────
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, fmt=".2f",
            cmap="Blues", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("chart4_correlation_heatmap.png", dpi=150)
plt.show()
print("Chart 4 done ✅")

# ── Prepare Data for ML ───────────────────────────
print("\nPreparing data for Machine Learning...")

# Encode yes/no columns
le = LabelEncoder()
yes_no_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning",
               "prefarea", "furnishingstatus"]

for col in yes_no_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

print("Features:", X.columns.tolist())
print("Target: price")

# ── Split Data ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ── Train Model ───────────────────────────────────
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully ✅")

# ── Make Predictions ──────────────────────────────
y_pred = model.predict(X_test)

# ── Evaluate Model ────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n── Model Performance ──────────────────────")
print(f"MAE  (Mean Absolute Error) : {mae:,.0f}")
print(f"RMSE (Root Mean Sq Error)  : {rmse:,.0f}")
print(f"R²   (Accuracy Score)      : {r2:.4f} ({r2*100:.2f}%)")

# ── Chart 5: Actual vs Predicted ──────────────────
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6,
            color="#2E75B6", edgecolors="white", linewidth=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2, label="Perfect Prediction")
plt.title("Actual vs Predicted House Prices",
          fontsize=16, fontweight="bold")
plt.xlabel("Actual Price", fontsize=13)
plt.ylabel("Predicted Price", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("chart5_actual_vs_predicted.png", dpi=150)
plt.show()
print("Chart 5 done ✅")

# ── Chart 6: Feature Importance ───────────────────
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

colors = ["#4CAF50" if c > 0 else "#FF7043"
          for c in feature_importance["Coefficient"]]
sns.barplot(x="Coefficient", y="Feature",
            data=feature_importance, palette=colors)
plt.title("Feature Importance (Model Coefficients)",
          fontsize=16, fontweight="bold")
plt.xlabel("Coefficient Value", fontsize=13)
plt.ylabel("Feature", fontsize=13)
plt.tight_layout()
plt.savefig("chart6_feature_importance.png", dpi=150)
plt.show()
print("Chart 6 done ✅")

print("\n🎉 All charts saved and ML model complete!")
print(f"\n📊 Final Model Accuracy: {r2*100:.2f}%")