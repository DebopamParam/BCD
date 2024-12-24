import markdown
import streamlit as st
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Architecture", page_icon="🤖")

st.title("Model Training")

markdown_text = """
    ### ***[Go to Kaggle to See and Run training Code](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)***
"""

st.markdown(markdown_text)

try:
    model_path = "static/additional/training.html"
    if os.path.exists(model_path):
        with open(model_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Calculate responsive height based on viewport
        components.html(html_content, height=2000, scrolling=True)
    else:
        st.error(
            "Model file not found. Please ensure 'static/additional/model.html' exists in the application directory."
        )
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
