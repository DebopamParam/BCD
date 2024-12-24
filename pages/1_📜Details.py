import streamlit as st

description = "static/additional/description.md"
# Read the markdown file
with open(description, "r", encoding="utf-8") as file:
    markdown_content = file.read()

# Display the markdown content
st.markdown(markdown_content, unsafe_allow_html=True)
