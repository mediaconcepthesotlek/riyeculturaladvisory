import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import json
import matplotlib.pyplot as plt

action_model = joblib.load("advisory_action_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
imputer = joblib.load("numeric_imputer.pkl")
le_focus = joblib.load("focus_encoder.pkl")
le_action = joblib.load("action_encoder.pkl")

# -------------------------
# Load label mappings
# -------------------------
with open("label_mappings.json", "r", encoding="utf-8") as f:
    label_mappings = json.load(f)

# -------------------------
# Load models & preprocessing objects
# -------------------------
focus_model = joblib.load("cultural_focus_model.pkl")
action_model = joblib.load("advisory_action_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
imputer = joblib.load("numeric_imputer.pkl")
le_focus = joblib.load("focus_encoder.pkl")
le_action = joblib.load("action_encoder.pkl")

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Cultural Advisory System",
    page_icon="🖼️",
    layout="wide"
)


st.markdown("""
# 🖼️ AI-Based Cultural Advisory System
Welcome! This system predicts **Cultural Focus** and provides actionable **Advisory Actions** based on your cultural and geographic data. Fill in the details below to get personalized advice.
""")

# -------------------------
# User Inputs
# -------------------------

with st.expander("ℹ️ About this App", expanded=False):
    st.write("""
    This tool uses AI to analyze cultural and geographic information and provide tailored advice for cultural engagement, preservation, or intervention. Enter as much detail as possible for best results.
    """)

st.header("1️⃣ Enter Cultural & Geographic Data")


with st.expander("Cultural Descriptions (Text)", expanded=True):
    text_inputs = {}
    text_columns = ['food', 'clothing', 'dance', 'religion', 'festivals', 'music_instruments']
    col1, col2, col3 = st.columns(3)
    for idx, col in enumerate(text_columns):
        example = label_mappings.get(col, {}).get("0", "")
        col_streamlit = [col1, col2, col3][idx % 3]
        with col_streamlit:
            text_inputs[col] = st.text_area(
                f"{col.replace('_', ' ').capitalize()} (comma-separated)",
                value="",
                help=f"Example: {example}" if example else None
            )

st.markdown("---")
st.header("2️⃣ Select Location")
lga_options = [(int(k), v) for k, v in label_mappings["lga"].items()]
lga_options.sort(key=lambda x: x[0])
lga_names = [v for _, v in lga_options]
lga_label_name = st.selectbox("LGA", lga_names)
lga_label = next(k for k, v in lga_options if v == lga_label_name)

town_wards_options = [(int(k), v) for k, v in label_mappings["town_wards"].items()]
town_wards_options.sort(key=lambda x: x[0])
town_wards_names = [v for _, v in town_wards_options]
town_wards_label_name = st.selectbox("Town/Wards", town_wards_names)
town_wards_label = next(k for k, v in town_wards_options if v == town_wards_label_name)

col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input("Latitude", format="%.6f", key="latitude_input")
with col2:
    longitude = st.number_input("Longitude", format="%.6f", key="longitude_input")

latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_input(text_inputs, lga, ward, lat, lon):
    # Combine text fields
    combined_text = " ".join([text_inputs[c] for c in text_columns])
    
    # TF-IDF transform
    text_features = tfidf.transform([combined_text])
    
    # Numeric features (imputed)
    numeric_features = np.array([[lga, ward, lat, lon]])
    numeric_features = imputer.transform(numeric_features)
    
    # Combine text + numeric
    return hstack([text_features, numeric_features])

# --- Customizable Advice Type ---
advice_types = ["Auto (Model Recommendation)", "Preserve", "Promote", "Intervene", "Educate"]
selected_advice_type = st.selectbox(
    "Select the type of advice you want:",
    advice_types,
    help="Choose 'Auto' to use the model's recommendation, or pick a specific type for tailored advice."
)

# -------------------------
# Prediction
# -------------------------

if st.button("Get Advisory"):
    X_input = preprocess_input(text_inputs, lga_label, town_wards_label, latitude, longitude)

    # Predict Cultural Focus
    focus_encoded = focus_model.predict(X_input)[0]
    focus_proba = focus_model.predict_proba(X_input)[0]
    focus_label = le_focus.inverse_transform([focus_encoded])[0]
    focus_conf = focus_proba.max()

    # Predict Advisory Action
    action_encoded = action_model.predict(X_input)[0]
    action_proba = action_model.predict_proba(X_input)[0]
    action_label = le_action.inverse_transform([action_encoded])[0]
    action_conf = action_proba.max()

    # Use selected advice type if not Auto
    if selected_advice_type != "Auto (Model Recommendation)":
        advice_key = selected_advice_type
    else:
        advice_key = action_label

    st.markdown("---")
    st.header("3️⃣ Advisory Results")

    st.success(f"**Cultural Focus:** {focus_label} (Confidence: {focus_conf:.2f})")
    st.info(f"**Recommended Advisory Action:** {advice_key} (Confidence: {action_conf:.2f})")

    # More detailed and dynamic advice
    detailed_advisory = {
        "Preserve": f"The system has identified that the main cultural focus is **{focus_label}**. To preserve this aspect, you should consider organizing local festivals, documenting oral histories, supporting artisans, and collaborating with elders to ensure traditions are passed down. Engage youth through workshops and digital storytelling.",
        "Promote": f"Since **{focus_label}** is prominent, promote it by hosting exhibitions, food fairs, and music events. Use social media campaigns, partner with schools for cultural days, and encourage local businesses to showcase related products. Highlight success stories and community impact.",
        "Intervene": f"There may be risks to **{focus_label}** in this area. Intervene by identifying at-risk practices, providing resources to practitioners, and working with local leaders to address challenges. Consider grant programs, awareness campaigns, and partnerships with NGOs.",
        "Educate": f"Education is key for sustaining **{focus_label}**. Develop school programs, community classes, and public talks. Create educational materials, invite cultural experts, and use interactive media to engage all age groups.",
    }
    advice_text = detailed_advisory.get(advice_key, f"Engage with local stakeholders for tailored advice on {focus_label} based on the predicted action: {advice_key}.")
    st.write(f"**Actionable Advice:** {advice_text}")

    # Add more explanation for the predictions
    with st.expander("How were these predictions made?", expanded=False):
        st.markdown(f"""
        - The **Cultural Focus** is predicted based on the combination of your text and location inputs. The model analyzes keywords and patterns to determine which cultural aspect is most relevant.
        - The **Advisory Action** is chosen by evaluating the predicted focus and your geographic context, suggesting the best next step for cultural engagement or preservation.
        - Confidence scores show how certain the model is about each prediction. If the confidence is low, try providing more detailed or varied input.
        """)

    with st.expander("Show Prediction Probabilities & Visualizations", expanded=False):
        st.subheader("Prediction Probabilities")
        # Cultural Focus Probabilities
        focus_df = pd.DataFrame({
            "Cultural Focus": le_focus.inverse_transform(np.arange(len(focus_proba))),
            "Probability": focus_proba
        }).sort_values(by="Probability", ascending=False)
        st.bar_chart(focus_df.set_index("Cultural Focus"))

        # Pie chart for cultural focus
        st.subheader("Cultural Focus Distribution (Pie Chart)")
        fig, ax = plt.subplots(figsize=(4, 4))
        focus_df.set_index("Cultural Focus").plot.pie(y="Probability", autopct="%.1f%%", legend=False, ylabel="", ax=ax)
        st.pyplot(fig)

        # Advisory Action Probabilities
        action_df = pd.DataFrame({
            "Advisory Action": le_action.inverse_transform(np.arange(len(action_proba))),
            "Probability": action_proba
        }).sort_values(by="Probability", ascending=False)
        st.bar_chart(action_df.set_index("Advisory Action"))
