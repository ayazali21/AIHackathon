import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Function to preprocess data and encode categorical features
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Select required columns
    selected_columns = ["Month", "Year", "Medicine Name", "Category", "Dosage (mg/ml)", 
                        "Hospital Name", "City", "Supplier Name", "Price per Unit (AED)", "Consumption (Units)"]
    df_filtered = df[selected_columns]
    
    # Label encode categorical columns
    label_columns = ["Medicine Name", "Category", "Hospital Name", "City", "Supplier Name"]
    label_encoders = {}

    for col in label_columns:
        le = LabelEncoder()
        df_filtered[col] = le.fit_transform(df_filtered[col])
        label_encoders[col] = le  # Store encoder for later use

    return df_filtered, label_encoders

# Train model and save encoders
def train_model(df, label_encoders):
    features = ["Month", "Year", "Medicine Name", "Category", "Dosage (mg/ml)", 
                "Hospital Name", "City", "Supplier Name", "Price per Unit (AED)"]
    target = "Consumption (Units)"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Save model and encoders
    joblib.dump(model, "medicine_model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

    return model, mae

# Load model and encoders
def load_model_and_encoders():
    model = joblib.load("medicine_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, label_encoders

# Streamlit UI for training
def train_ui():
    st.title("Train Medicine Consumption Model")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df, label_encoders = load_data(uploaded_file)

        if st.button("Train Model"):
            model, mae = train_model(df, label_encoders)
            st.success(f"Model trained successfully! MAE: {mae:.2f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

            # Feature Importance
            feature_importance = pd.DataFrame({
                "Feature": df.columns[:-1],  # Exclude target column
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.subheader("Feature Importance")
            st.bar_chart(feature_importance.set_index("Feature"))

            # # Display model parameters
            # st.subheader("Model Parameters")
            # st.json(model.get_params())

# Streamlit UI for prediction
def predict_ui():
    st.title("Medicine Consumption Prediction")

    try:
        model, label_encoders = load_model_and_encoders()
    except:
        st.error("Please train the model first.")
        return

    st.header("Predict Consumption")

    month = st.selectbox("Month", list(range(1, 13)))
    year = st.selectbox("Year", list(range(2024, 2027)))

    # Allow user to select medicines dynamically
    medicine_names = list(label_encoders["Medicine Name"].classes_)
    selected_medicines = st.multiselect("Select Medicines", medicine_names)

    if st.button("Predict Consumption"):
        predictions = []

        for medicine in selected_medicines:
            if medicine not in medicine_names:
                st.error(f"Medicine '{medicine}' not found in training data.")
                return

            encoded_medicine = label_encoders["Medicine Name"].transform([medicine])[0]

            # Construct a DataFrame for prediction
            df_predict = pd.DataFrame({
                "Month": [month],
                "Year": [year],
                "Medicine Name": [encoded_medicine],
                "Category": [0],  # Replace with actual category encoding if needed
                "Dosage (mg/ml)": [500],  # Replace with correct value
                "Hospital Name": [0],
                "City": [0],
                "Supplier Name": [0],
                "Price per Unit (AED)": [10.0]
            })

            # Make predictions
            prediction = model.predict(df_predict)
            predictions.append({"Medicine": medicine, "Predicted Consumption": prediction[0]})

        # Display predictions
        pred_df = pd.DataFrame(predictions)
        st.subheader("Predicted Consumption by Medicine")
        st.bar_chart(pred_df.set_index("Medicine")["Predicted Consumption"])

# Main Streamlit app
def main():
    option = st.selectbox("Select Option", ["Train Model", "Predict Consumption"])

    if option == "Train Model":
        train_ui()
    else:
        predict_ui()

if __name__ == "__main__":
    main()
