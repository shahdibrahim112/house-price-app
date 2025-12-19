import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("train.csv")

# Feature engineering
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF']
features = ['OverallQual','GrLivArea','TotalArea','HouseAge']
X = df[features]
y = np.log1p(df['SalePrice'])

# Train model
model = LinearRegression()
model.fit(X, y)

# GUI
st.title("🏡 House Price Predictor")
overall_qual = st.number_input("Overall Quality", min_value=1, max_value=10, value=5)
gr_liv_area = st.number_input("Living Area (sq ft)", value=1000)
# ... باقي الـ features

if st.button("Predict Price"):
    X_input = np.array([[overall_qual, gr_liv_area, ...]])
    pred = model.predict(X_input)
    price = np.expm1(pred)[0]
    st.success(f"Predicted Price: ${price:,.2f}")