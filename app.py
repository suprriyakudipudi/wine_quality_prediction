import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🍷 Wine Quality Prediction App")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("winequality-red.csv")
    return data

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# Features & Target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy:.2f}")

# Sidebar for user input
st.sidebar.header("Enter Wine Chemical Properties")

def user_input():
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 15.0, 1.9)
    chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.076)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1, 70, 11)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6, 300, 34)
    density = st.sidebar.slider("Density", 0.990, 1.005, 0.9978)
    pH = st.sidebar.slider("pH", 2.5, 4.5, 3.51)
    sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.56)
    alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 9.4)

    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Prediction
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

st.subheader("Predicted Wine Quality")
st.info(f"🍾 Predicted Quality Score: {prediction[0]}")
