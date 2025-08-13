#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os

st.title("NFL Over/Under Predictions – 2025")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Week selector ---
# List all prediction CSVs in the folder
prediction_files = [f for f in os.listdir(BASE_DIR) if f.endswith("_predictions.csv")]

# Extract week numbers from filenames like "week1_2025_predictions.csv"
weeks_available = sorted([
    int(f.split("_")[0].replace("week", ""))
    for f in prediction_files
])

# Let user pick the week
selected_week = st.selectbox("Select Week:", weeks_available)

# Build file paths for selected week
pred_file = f"week{selected_week}_2025_predictions.csv"

# Load main predictions table
preds = pd.read_csv(os.path.join(BASE_DIR, pred_file))

# Display each game and its neighbors
for i, row in preds.iterrows():
    st.markdown(
        f"**{row['Game']}** | Spread: {row['Spread']:.1f} | Total: {row['Total']:.1f} "
        f"| **Prediction:** {row['Prediction']}"
    )
    st.write(
        f"Confidence %: {row['ConfidencePercent']*100:.1f}% "
        f"| Avg Distance: {row['AvgDistance']} "
        f"| Score: {row['ConfidenceScore']:.3f}"
    )

    # Load neighbors for this game
    neighbors_file = f"neighbors_{i+1}_week{selected_week}.csv"
    neighbors_path = os.path.join(BASE_DIR, neighbors_file)

    if os.path.exists(neighbors_path):
        neighbors = pd.read_csv(neighbors_path)
        st.dataframe(neighbors.round(3))
    else:
        st.warning(f"Neighbors file not found: {neighbors_file}")
