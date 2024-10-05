import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_layer_sizes[0]))
        layers.append(nn.SiLU())  # Adding ReLU activation function
        
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.3))
        
        layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Initialize and load the model
model = Model(18, 1, [64, 32, 16])  # Make sure the dimensions match your architecture
model.load_state_dict(torch.load("model.pth"))

# Define the path where the label encoders are saved
label_encoders_path = 'label_encoders'  # Adjust path if necessary

# Load the label encoders
loaded_label_encoders = {}
for col in ['body_type', 'sex', 'diet', 'shower_freq', 'heat_energy_source', 'transport', 'social',
            'air_travel_freq', 'waste_bag_size', 'energy_eff', "recycling", "cooking_with"]:
    encoder_file = os.path.join(label_encoders_path, f"{col}.pkl")
    if os.path.exists(encoder_file):
        loaded_label_encoders[col] = joblib.load(encoder_file)

# Load the scaler
scaler_file = 'scaler.pkl'  # Update the scaler file path as necessary
scaler = joblib.load(scaler_file) if os.path.exists(scaler_file) else None

# Streamlit app title
st.title("Carbon Emission Calculator")

# Collect user inputs
body_type = st.selectbox("Body Type", ['overweight', 'obese', 'underweight', 'normal'])
sex = st.selectbox("Sex", ["male", "female"])
diet = st.selectbox("Diet", ['pescatarian', 'vegetarian', 'omnivore', 'vegan'])
shower_freq = st.selectbox("Shower Frequency", ['daily', 'less frequently', 'more frequently', 'twice a day'])
heat_energy_source = st.selectbox("Heat Energy Source", ['coal', 'natural gas', 'wood', 'electricity'])
transport = st.selectbox("Transport", ['public', 'walk/bicycle', 'private'])
social = st.selectbox("Social Interaction Frequency", ['often', 'never', 'sometimes'])
monthly_grocery_bill = st.number_input("Monthly Grocery Bill ($)", min_value=0.0)
air_travel_freq = st.selectbox("Air Travel Frequency", ['frequently', 'rarely', 'never', 'very frequently'])
vehicle_km = st.number_input("Vehicle KM", min_value=0)
waste_bag_size = st.selectbox("Waste Bag Size", ['large', 'extra large', 'small', 'medium'])
waste_bag_count = st.number_input("Waste Bag Count", min_value=0)
tv_daily_time = st.number_input("TV Daily Time (hours)", min_value=0.0, max_value=24.0)
monthly_clothes = st.number_input("Pices of clothes", min_value=0.0, max_value=50.0)
internet_daily_time = st.number_input("Internet Daily Time (hours)", min_value=0.0, max_value=24.0)
energy_eff = st.selectbox("Energy Efficiency", ['No', 'Sometimes', 'Yes'])
recycling = st.multiselect("Recycling", ["Metal", "Glass", "Paper", "Plastic"], default=[])
cooking_with = st.multiselect("Cooking With", ['Airfryer', 'Grill', 'Microwave', 'Oven', 'Stove'], default=[])

# Button to calculate
if st.button("Calculate Carbon Emission"):
    # Prepare the input features as a DataFrame
    input_data = pd.DataFrame({
        'body_type': [body_type],
        'sex': [sex],
        'diet': [diet],
        'shower_freq': [shower_freq],
        'heat_energy_source': [heat_energy_source],
        'transport': [transport],
        'social': [social],
        'monthly_grocery_bill': [monthly_grocery_bill],
        'air_travel_freq': [air_travel_freq],
        'vehicle_km': [vehicle_km],
        'waste_bag_size': [waste_bag_size],
        'waste_bag_count': [waste_bag_count],
        'tv_daily_time': [tv_daily_time],
        'monthly_clothes': [monthly_clothes],
        'internet_daily_time': [internet_daily_time],
        'energy_eff': [energy_eff],
        'recycling': [str(sorted(list(recycling)))],  
        'cooking_with': [str(sorted(list(cooking_with)))] 
    })

    # Apply label encoders to categorical columns
    for col in loaded_label_encoders.keys():
        if col in input_data.columns:
            input_data[col] = loaded_label_encoders[col].transform(input_data[col])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Convert the scaled input data to a PyTorch tensor
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)


    # Calculate carbon emission using the model
    with torch.no_grad():  # Disable gradient calculation for inference
        carbon_emission = model.forward(input_tensor).numpy()  # Convert the output tensor to numpy
    
    value = np.exp(carbon_emission[0][0])

    # Display the result
    st.write(f"Estimated Carbon Emission: {value} kg CO2")

