import streamlit as st
import pandas as pd
import pickle

# Load model

with open('model.pkl', 'rb') as file:
  model = pickle.load(file)

st.set_page_config(page_title="BigMart Sales Prediction", page_icon="📈")

st.title("📈 BigMart Sales Prediction")
st.write("Enter product and outlet information to predict sales.")

# Numerical inputs

item_weight = st.number_input("Item Weight", min_value=0.0, value=12.0)
item_visibility = st.number_input("Item Visibility", min_value=0.0, value=0.05)
item_mrp = st.number_input("Item MRP", min_value=0.0, value=150.0)
outlet_establishment_year = st.number_input(
"Outlet Establishment Year",
min_value=1980,
max_value=2025,
value=2000
)

# Encoded categorical inputs

st.subheader("Encoded Categorical Values")

item_fat_content = st.number_input(
"Item Fat Content (Encoded)",
min_value=0,
value=0
)

item_type = st.number_input(
"Item Type (Encoded)",
min_value=0,
value=0
)

outlet_size = st.number_input(
"Outlet Size (Encoded)",
min_value=0,
value=0
)

outlet_location_type = st.number_input(
"Outlet Location Type (Encoded)",
min_value=0,
value=0
)

outlet_type = st.number_input(
"Outlet Type (Encoded)",
min_value=0,
value=0
)

if st.button("Predict Sales"):

    data = pd.DataFrame({
        'Item_Weight': [item_weight],
        'Item_Fat_Content': [item_fat_content],
        'Item_Visibility': [item_visibility],
        'Item_Type': [item_type],
        'Item_MRP': [item_mrp],
        'Outlet_Establishment_Year': [outlet_establishment_year],
        'Outlet_Size': [outlet_size],
        'Outlet_Location_Type': [outlet_location_type],
        'Outlet_Type': [outlet_type]
    })

    prediction = model.predict(data)

    st.success(
        f"Predicted Item Outlet Sales: ${prediction[0]:,.2f}"
    )

