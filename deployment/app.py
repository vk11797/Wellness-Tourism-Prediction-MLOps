import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Wellness Tourism Predictor", layout="wide")

# --- 2. LOAD THE MODEL PIPELINE ---
@st.cache_resource
def load_model():
    # Downloads the latest model from your Hugging Face Model Hub
    repo_id = "VKblues2025/wellness-tourism-model"
    filename = "model.joblib"

    # Show a spinner while downloading to look professional
    with st.spinner("Initializing predictive engine..."):
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return joblib.load(model_path)

try:
    model = load_model()
    # SUCCESS MESSAGE REMOVED: (Uncomment the line below if you ever need to debug)
    # st.success("✅ Model Pipeline Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- 3. UI HEADER ---
st.title("🏨 Wellness Tourism Package Prediction")
st.markdown("Enter customer details in the sidebar to predict package purchase probability.")

# --- 4. USER INPUTS (SIDEBAR) ---
st.sidebar.header("Customer Information")

def get_user_input():
    # Primary features for the user to adjust
    occupation = st.sidebar.selectbox("Occupation", ['Salaried', 'Small Business', 'Free Lancer', 'Large Business'])
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Unmarried', 'Divorced'])
    designation = st.sidebar.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
    age = st.sidebar.slider("Age", 18, 70, 30)
    duration_of_pitch = st.sidebar.number_input("Duration of Pitch (mins)", 0, 120, 15)
    monthly_income = st.sidebar.number_input("Monthly Income", 0, 100000, 25000)
    number_of_trips = st.sidebar.slider("Number of Trips", 1, 20, 2)
    passport = st.sidebar.selectbox("Has Passport?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    # --- DUMMY COLUMNS ---
    # These match the exact schema the model was trained on to prevent 'ValueError'
    data = {
        'Age': [age],
        'Occupation': [occupation],
        'Gender': [gender],
        'DurationOfPitch': [duration_of_pitch],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [number_of_trips],
        'MonthlyIncome': [monthly_income],
        'Passport': [passport],
        'Designation': [designation],

        # Placeholder columns required by the Scikit-learn Pipeline
        'Unnamed: 0': [0],
        'TypeofContact': ['Self Enquiry'],
        'NumberOfFollowups': [0],
        'NumberOfChildrenVisiting': [0],
        'ProductPitched': ['Deluxe'],
        'OwnCar': [0],
        'CityTier': [1],
        'PreferredPropertyStar': [3],
        'CustomerID': [0],
        'PitchSatisfactionScore': [3],
        'NumberOfPersonVisiting': [1]
    }
    return pd.DataFrame(data)

input_df = get_user_input()

# --- 5. PREDICTION LOGIC ---
st.subheader("Customer Profile Summary")
# Show only the meaningful columns to the user
st.write(input_df[['Age', 'Occupation', 'Gender', 'Designation', 'MonthlyIncome', 'Passport']])

# Predict Button
if st.button("Predict"):
    try:
        # Pass the raw dataframe to the pipeline (handles scaling/encoding automatically)
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        st.divider()

        # Display specific requested results
        if prediction[0] == 1:
            st.warning(f"🎯 **High Probability: This customer is likely to purchase!**")
            st.write(f"Confidence Score: {probability:.2%}")
        else:
            st.info(f"😴 **Low Probability: This customer is unlikely to purchase.**")
            st.write(f"Confidence Score: {probability:.2%}")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Developed by VK using MLOps Best Practices.")
