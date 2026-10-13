# Save as app.py and run via: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Used Car Valuation Tool", layout="centered")

# Simple light color scheme
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f7fa;
        }
        section[data-testid="stSidebar"] {
            background-color: #eef1f6;
        }
        div.stButton > button {
            background-color: #cfe8f3;
            color: #1a1a1a;
            border: 1px solid #a9c9d9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Used-Car price model predictor")
st.markdown("Enter vehicle specifications below to estimate fair market value.")

st.sidebar.header("Vehicle Specifications")

# Inputs corresponding to SelectKBest Top 12 features
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=30.0, max_value=600.0, value=85.0, step=5.0)
engine = st.sidebar.number_input("Engine Displacement (CC)", min_value=600, max_value=6000, value=1248, step=50)
vehicle_age = st.sidebar.slider("Vehicle Age (Years)", min_value=0, max_value=25, value=5)
mileage = st.sidebar.number_input("Fuel Efficiency (km/l)", min_value=5.0, max_value=40.0, value=18.5, step=0.5)
seats = st.sidebar.selectbox("Seating Capacity", [2, 4, 5, 6, 7, 8], index=2)

fuel_type = st.sidebar.selectbox("Fuel Type", ["Diesel", "Petrol", "Other"])
transmission = st.sidebar.radio("Transmission Type", ["Manual", "Automatic"])
brand = st.sidebar.selectbox("Vehicle Brand", ["Toyota", "Maruti", "BMW", "Mercedes-Benz", "Other"])

# Construct inference feature vector matching Section 5.2 feature sequence
input_data = {
    'max_power': max_power,
    'engine': engine,
    'vehicle_age': vehicle_age,
    'fuel_type_Diesel': 1 if fuel_type == "Diesel" else 0,
    'transmission_type_Manual': 1 if transmission == "Manual" else 0,
    'fuel_type_Petrol': 1 if fuel_type == "Petrol" else 0,
    'brand_Toyota': 1 if brand == "Toyota" else 0,
    'seats': seats,
    'brand_Maruti': 1 if brand == "Maruti" else 0,
    'mileage': mileage,
    'brand_BMW': 1 if brand == "BMW" else 0,
    'brand_Mercedes-Benz': 1 if brand == "Mercedes-Benz" else 0
}

input_df = pd.DataFrame([input_data])

# Specification summary
st.subheader("Specification Summary")
summary = {
    "Brand": brand,
    "Fuel Type": fuel_type,
    "Transmission": transmission,
    "Vehicle Age (years)": vehicle_age,
    "Engine (CC)": engine,
    "Max Power (bhp)": max_power,
    "Mileage (km/l)": mileage,
    "Seats": seats
}
st.table(pd.DataFrame(summary.items(), columns=["Specification", "Value"]))

if st.button("Calculate Estimated Price"):
    # Mock inference logic: prediction = expm1(model.predict(input_df)[0])
    # Assuming trained model object 'gb_model' is loaded
    estimated_log_price = gb_model.predict(input_df)[0]
    estimated_price = np.expm1(estimated_log_price)
    
    st.subheader(f"Estimated Market Valuation: ₹{estimated_price:,.2f}")
    st.info("Valuation calculated using Section 5.2 SelectKBest Gradient Boosting Model Pipeline.")