import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

# ======================
# 1️⃣ Load Data
# ======================
df = pd.read_csv("train.csv")
print("=== Data Preview ===")
print(df.head())
print("\n=== Shape of Data ===")
print(df.shape)
print("\n=== Data Info ===")
df.info()

# ======================
# 2️⃣ Missing Values
# ======================
missing = df.isnull().sum().sort_values(ascending=False)
print("\n=== Top 10 Missing Values ===")
print(missing.head(10))

plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# ======================
# 3️⃣ Handle Missing Values
# ======================
num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum().max())

# ======================
# 4️⃣ Outliers Visualization
# ======================
plt.figure(figsize=(12,5))
sns.boxplot(x=df['GrLivArea'])
plt.title("Boxplot of GrLivArea")
plt.show()

plt.figure(figsize=(12,5))
sns.histplot(df['GrLivArea'], bins=30, kde=True)
plt.title("Histogram of GrLivArea")
plt.show()

# ======================
# 5️⃣ Remove Outliers
# ======================
Q1 = df['GrLivArea'].quantile(0.25)
Q3 = df['GrLivArea'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df = df[(df['GrLivArea'] >= lower) & (df['GrLivArea'] <= upper)]
print("\n=== Shape After Removing Outliers ===")
print(df.shape)

# ======================
# 6️⃣ Feature Engineering
# ======================
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF']
print(df[['HouseAge','TotalArea']].head())

# ======================
# 7️⃣ Feature Visualizations
# ======================
features = ['OverallQual','GrLivArea','TotalArea','HouseAge']

for col in features:
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"Histogram of {col}")
    
    plt.subplot(1,3,2)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    
    plt.subplot(1,3,3)
    sns.scatterplot(x=df[col], y=df['SalePrice'])
    plt.title(f"{col} vs SalePrice")
    
    plt.tight_layout()
    plt.show()

# ======================
# 8️⃣ Correlation Heatmap
# ======================
plt.figure(figsize=(10,8))
sns.heatmap(df[features + ['SalePrice']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# ======================
# 9️⃣ Feature Selection & Split
# ======================
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 🔟 Model Training
# ======================
model = LinearRegression()
model.fit(X_train, y_train)

# ======================
# 1️⃣1️⃣ Evaluation
# ======================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² (Accuracy): {r2:.2f}")
features = ['OverallQual','GrLivArea','TotalArea','HouseAge',
            'GarageCars','GarageArea','TotalBsmtSF','FullBath','HalfBath','Fireplaces']
X = df[features]

# Log-transform the target
y = np.log1p(df['SalePrice'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Revert log-transform to get real values
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

# Evaluate
mae = mean_absolute_error(y_test_exp, y_pred_exp)
rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
r2 = r2_score(y_test_exp, y_pred_exp)

print("=== Model Evaluation After Feature Addition & Log Transform ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² (Accuracy): {r2:.2f}")