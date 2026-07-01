import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Raisin Class Predictor",
    page_icon="🍇",
    layout="centered"
)

st.title("🍇 Raisin Classification App")
st.write("Predict whether a raisin is **Kecimen** or **Besni** based on its physical characteristics.")

# ----------------------------------------------------------------------
# Data Loading & Model Training (Cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_and_train_model():
    # Load dataset
    try:
        df = pd.read_csv("Raisin_Dataset.csv")
    except FileNotFoundError:
        st.error("Please ensure 'Raisin_Dataset.csv' is in the same directory as this script.")
        st.stop()
        
    # Map target class to numeric values
    df["Class"] = df["Class"].map({"Kecimen": 1, "Besni": 0})
    
    # Split features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.1
    )
    
    # Define and fit model matching your notebook parameters
    model = RandomForestClassifier(
        n_estimators=10,
        criterion="gini",
        max_depth=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Calculate test accuracy
    test_acc = model.score(X_test, y_test)
    
    return model, test_acc, X

# Initialize model and configurations
model, test_accuracy, X_features = load_and_train_model()

# ----------------------------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------------------------
st.sidebar.header("🔧 Input Raisin Features")

def user_input_features():
    # Use standard default values close to the dataset distribution
    area = st.sidebar.number_input("Area", min_value=0, max_value=300000, value=80000)
    major_axis = st.sidebar.number_input("MajorAxisLength", min_value=0.0, max_value=1000.0, value=400.0)
    minor_axis = st.sidebar.number_input("MinorAxisLength", min_value=0.0, max_value=1000.0, value=250.0)
    eccentricity = st.sidebar.slider("Eccentricity", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    convex_area = st.sidebar.number_input("ConvexArea", min_value=0, max_value=300000, value=85000)
    extent = st.sidebar.slider("Extent", min_value=0.0, max_value=1.0, value=0.70, step=0.01)
    perimeter = st.sidebar.number_input("Perimeter", min_value=0.0, max_value=3000.0, value=1100.0)
    
    data = {
        "Area": area,
        "MajorAxisLength": major_axis,
        "MinorAxisLength": minor_axis,
        "Eccentricity": eccentricity,
        "ConvexArea": convex_area,
        "Extent": extent,
        "Perimeter": perimeter
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ----------------------------------------------------------------------
# Main Panel Output
# ----------------------------------------------------------------------

# Display model metrics briefly
st.sidebar.markdown("---")
st.sidebar.metric(label="Model Test Accuracy", value=f"{test_accuracy:.2%}")

# Display chosen inputs
st.subheader("Selected Specifications")
st.dataframe(input_df)

# Prediction button
if st.button("🔮 Predict Class", type="primary"):
    # Re-order columns to match the training data format exactly
    input_df = input_df[X_features.columns]
    
    # Perform prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    st.markdown("---")
    st.subheader("Result")
    
    if prediction == 1:
        st.success("🎉 The raisin is classified as: **Kecimen**")
    else:
        st.info("🎉 The raisin is classified as: **Besni**")
        
    # Probability chart
    st.write("### Prediction Confidence")
    prob_df = pd.DataFrame({
        "Class": ["Besni", "Kecimen"],
        "Probability": [prediction_proba[0], prediction_proba[1]]
    })
    st.bar_chart(data=prob_df, x="Class", y="Probability")