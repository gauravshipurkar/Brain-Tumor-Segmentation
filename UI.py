import streamlit as st
from PIL import Image
# MAIN FUNCTION:
# ----------------------------------------------------
st.set_page_config(layout="wide")

st.write(""" # 3D Brain Tumor Image Segmentation:
Please upload the required image. """)

uploaded_file = st.sidebar.file_uploader(
    "Choose a Brain Tumor 3D Image:", type=["png", "jpg"])

if uploaded_file is not None:
    if st.sidebar.button('Get Results..'):

        image = Image.open(uploaded_file)
        st.image(image)
