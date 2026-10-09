import streamlit as st
import pandas as pd
import numpy as np
import joblib
from agent_reasoning import generate_agent_explanation

# Page setup
st.set_page_config(
    page_title="Advanced ML Property Valuation Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .report-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .price-usd {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .price-inr {
        font-size: 1.3rem;
        color: #cbd5e1;
        margin-bottom: 12px;
    }
    .bound-container {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("property_valuation_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model artifact: {e}")
    st.stop()

USD_TO_INR = 83.50

def format_inr(number):
    s = str(int(number))
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    parts = []
    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.insert(0, remaining)
    return f"₹{','.join(parts)},{last_three}"

st.markdown('<div class="main-header">🏠 Advanced ML Property Valuation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deploying localized machine learning architectures for high-precision real estate appraisal.</div>', unsafe_allow_html=True)

col_input, col_output = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("📋 Property Specifications")
    
    st.markdown("##### 📐 Dimensions & Core Layout")
    col1, col2 = st.columns(2)
    with col1:
        area_sqft = st.number_input("Living Area (sqft)", min_value=300, max_value=15000, value=3500, step=50)
        bedrooms = st.radio("Bedrooms", options=[1, 2, 3, 4, 5, 6, 7, 8], index=2, horizontal=True)
        floors = st.number_input("Number of Floors", min_value=1, max_value=5, value=2)
    with col2:
        lot_size = st.number_input("Total Lot Size (sqft)", min_value=500, max_value=50000, value=5000, step=100)
        bathrooms = st.radio("Bathrooms", options=[1, 2, 3, 4, 5, 6], index=2, horizontal=True)
        fireplaces = st.slider("Fireplaces Count", 0, 5, 1)

    st.markdown("##### 🛠️ Age & Structural Quality")
    col3, col4 = st.columns(2)
    with col3:
        property_age = st.number_input("Property Age (Years)", min_value=0, max_value=120, value=5)
        renovation_status = st.slider("Renovation Tier (0=Original, 10=Modern)", 0, 10, 5)
    with col4:
        construction_quality = st.slider("Construction Build Quality (1-10)", 1, 10, 7)
        is_duplex = st.checkbox("Is this a Duplex property?", value=False)
        property_type_Duplex = 1 if is_duplex else 0

    st.markdown("##### 📍 Location & Environmental Amenities")
    col5, col6 = st.columns(2)
    with col5:
        neighborhood_score = st.slider("Neighborhood Rating (1-10)", 1, 10, 8)
        school_rating = st.slider("Public School Quality (1-10)", 1, 10, 7)
    with col6:
        water_supply_score = st.slider("Water Infra Reliability (1-10)", 1, 10, 9)
        green_space_index = st.slider("Parks & Greenery Proximity (1-10)", 1, 10, 8)

with col_output:
    st.subheader("📊 Valuation Output")
    st.write("Adjust the specification parameters on the left and trigger the engine to generate an assessment.")
    
    if st.button("Run Prediction Model", type="primary", use_container_width=True):
        features = [
            'area_sqft', 'bedrooms', 'bathrooms', 'floors', 'property_age',
            'renovation_status', 'lot_size', 'neighborhood_score', 'school_rating',
            'parking_availability', 'construction_quality', 'water_supply_score',
            'green_space_index', 'Fireplaces', 'property_type_Duplex'
        ]
        
        input_data = {
            'area_sqft': area_sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'floors': floors,
            'property_age': property_age,
            'renovation_status': renovation_status,
            'lot_size': lot_size,
            'neighborhood_score': neighborhood_score,
            'school_rating': school_rating,
            'parking_availability': 1,
            'construction_quality': construction_quality,
            'water_supply_score': water_supply_score,
            'green_space_index': green_space_index,
            'Fireplaces': fireplaces,
            'property_type_Duplex': property_type_Duplex
        }
        
        input_df = pd.DataFrame([input_data])[features]
        log_pred = model.predict(input_df)
        base_price_usd = float(np.expm1(log_pred)[0])
        
        mae_value = 18541.37
        lower_bound_usd = max(0, base_price_usd - mae_value)
        upper_bound_usd = base_price_usd + mae_value
        
        base_price_inr = base_price_usd * USD_TO_INR
        lower_bound_inr = lower_bound_usd * USD_TO_INR
        upper_bound_inr = upper_bound_usd * USD_TO_INR

        st.markdown(f"""
        <div class="report-card">
            <h4 style="color: #4ade80; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.05em;">Valuation Report Generated</h4>
            <span style="color: #94a3b8; font-size: 0.85rem;">POINT ESTIMATE MARKET VALUE</span>
            <div class="price-usd">${base_price_usd:,.2f}</div>
            <div class="price-inr">({format_inr(base_price_inr)})</div>
            <div class="bound-container">
                <div>
                    <span style="color: #94a3b8; font-size: 0.75rem;">CONSERVATIVE (-MAE)</span>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f87171; margin-top: 4px;">${lower_bound_usd:,.2f}</div>
                    <div style="font-size: 0.85rem; color: #fca5a5;">{format_inr(lower_bound_inr)}</div>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.75rem;">OPTIMISTIC (+MAE)</span>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #60a5fa; margin-top: 4px;">${upper_bound_usd:,.2f}</div>
                    <div style="font-size: 0.85rem; color: #93c5fd;">{format_inr(upper_bound_inr)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("⚠️ Conversions use a reference standard rate (1 USD ≈ ₹83.50).")

        # ------------------ Bedrock Agent Reasoning Layer ------------------
        with st.spinner("🤖 Agent analyzing valuation drivers & layout balance..."):
            specs_dict = {
                "living_area_sqft": area_sqft,
                "lot_size_sqft": lot_size,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "floors": floors,
                "property_age": property_age,
                "build_quality": construction_quality,
                "renovation_tier": renovation_status,
                "neighborhood_rating": neighborhood_score,
                "school_quality": school_rating,
                "parks_proximity": green_space_index,
                "is_duplex": is_duplex
            }
            agent_narrative = generate_agent_explanation(specs_dict, base_price_usd)

        st.markdown("### 💡 Agent Consultative Breakdown")
        st.info(agent_narrative)
        # -------------------------------------------------------------------
