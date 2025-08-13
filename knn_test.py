#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os

st.title("NFL Over/Under Predictions – 2025 Week 1")

# Load the main Week 1 predictions table
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
preds = pd.read_csv(os.path.join(BASE_DIR, "week1_2025_predictions.csv"))

# Display each game and its neighbors
for i, row in preds.iterrows():
    st.markdown(f"**{row['Game']}** | Spread: {row['Spread']:.1f} | Total: {row['Total']:.1f} | **Prediction:** {row['Prediction']}")
    st.write(f"Confidence %: {row['ConfidencePercent']*100:.1f}% | Avg Distance: {row['AvgDistance']} | Score: {row['ConfidenceScore']:.3f}")

    # Load and display neighbors for this game
    neighbors = pd.read_csv(os.path.join(BASE_DIR, f"neighbors_{i+1}.csv"))
    st.dataframe(neighbors.round(3))
