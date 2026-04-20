import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import requests
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="FMCG Sales Predictor",
    page_icon="🛒",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 20px; }
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        font-weight: bold;
        height: 50px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛍️ FMCG Sales Predictor")
st.markdown("---")

# Load and train model on the fly
@st.cache_resource
def load_and_train_model():
    try:
        # Try to load local file first
        if os.path.exists('cleaned_data.csv'):
            df = pd.read_csv('cleaned_data.csv')
        else:
            # If file doesn't exist, use the uploaded data from your screenshot
            # This is a sample that will be used if no CSV is found
            st.warning("⚠️ Using sample data. Upload 'cleaned_data.csv' for better predictions.")
            
            # Create sample data from what you showed earlier
            sample_data = """Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Size,Outlet_Location_Type,Outlet_Type,store_age,Item_Outlet_Sales
9.3,Low Fat,0.016047301,Dairy,249.8092,Medium,Tier 1,Supermarket Type1,26,3735.138
5.92,Regular,0.019278216,Soft Drinks,48.2692,Medium,Tier 3,Supermarket Type2,16,443.4228
17.5,Low Fat,0.016760075,Meat,141.618,Medium,Tier 1,Supermarket Type1,26,2097.27
19.2,Regular,0.0,Fruits and Vegetables,182.095,Small,Tier 3,Grocery Store,27,732.38
8.93,Low Fat,0.0,Household,53.8614,High,Tier 3,Supermarket Type1,38,994.7052"""
            df = pd.read_csv(StringIO(sample_data))
        
        # Preprocessing
        df['Item_Weight'] = df.groupby('Item_Type')['Item_Weight'].transform(lambda x: x.fillna(x.median()))
        df['Item_Weight'].fillna(df['Item_Weight'].median(), inplace=True)
        df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
        
        # Encode categorical variables
        categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Prepare features and target
        feature_cols = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 
                        'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'store_age']
        
        X = df[feature_cols]
        y = df['Item_Outlet_Sales']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        return model, label_encoders, feature_cols
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load model
with st.spinner("🔄 Loading model..."):
    model, label_encoders, feature_cols = load_and_train_model()
st.success("✅ Model ready!")

# Input form
st.markdown("### 📋 Enter Item Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        item_weight = st.number_input("Item Weight", min_value=0.0, value=12.5, step=0.1)
        item_fat_content = st.selectbox("Fat Content", ['Low Fat', 'Regular'])
        item_visibility = st.slider("Item Visibility", 0.0, 0.35, 0.05, 0.001)
        item_type = st.selectbox("Item Type", 
            ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
             'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Canned', 'Breads',
             'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Starchy Foods', 'Others'])
        item_mrp = st.number_input("Item MRP ($)", min_value=0.0, value=140.0, step=1.0)
    
    with col2:
        outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
        outlet_location = st.selectbox("Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.selectbox("Outlet Type", 
            ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])
        store_age = st.number_input("Store Age (years)", min_value=0, value=15, step=1)
    
    submitted = st.form_submit_button("🔮 Predict Sales")

if submitted:
    try:
        # Prepare input
        input_data = pd.DataFrame([[
            item_weight, item_fat_content, item_visibility, item_type,
            item_mrp, outlet_size, outlet_location, outlet_type, store_age
        ]], columns=['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
                     'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'store_age'])
        
        # Encode
        for col, le in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])
        
        input_data = input_data[feature_cols]
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Display
        st.markdown("---")
        st.markdown(f"""
            <div class="prediction-box">
                <h3 style='color: #666;'>Predicted Sales</h3>
                <h1 style='color: #ff4b4b; font-size: 48px;'>${prediction:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #999;'>🛒 FMCG Sales Predictor | Built with Streamlit</p>", unsafe_allow_html=True)