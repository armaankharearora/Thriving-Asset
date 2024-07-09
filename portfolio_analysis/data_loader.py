import streamlit as st
import pandas as pd

@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)
