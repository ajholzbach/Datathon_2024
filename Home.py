import streamlit as st
from PIL import Image

# Main app home page
st.set_page_config(page_title="Datathon 2024 Team Well, Well, Well", layout="wide")
st.title('Well, Well, Well')
st.markdown("## 2024 Datathon Submission, Chevron Track")
st.markdown("### Team Members: Alex Holzbach, Arnav Brurdgunte, Chuk Uzowihe, and Evan Stegall")
st.markdown("#### [Github Repo](https://github.com/ajholzbach/Datathon_2024)")
st.sidebar.header("Navigation")
wellwellwell = Image.open('Figures/wellwellwell.png')
st.image(wellwellwell, caption='Well, Well, Well', width=600)
