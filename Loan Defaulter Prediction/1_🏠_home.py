import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image

# --SETTING CONFIGS---
# Emojis - https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Loan Defaulter Predictor", page_icon=":computer:",layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# --ASSETS--
lottie_loan = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_apslgmwl.json")
image_loan_amt_hist = Image.open("images/loan_amt_hist.png")
image_int_rate_hist = Image.open("images/int_rate_hist.png")
image_correlation = Image.open("images/correlation.png")
image_violinplot_loanamt = Image.open("images/violinplot_loanamt.png")
image_corr_encoding = Image.open("images/corr_encoding.png")
image_smote = Image.open("images/smote.png")


title = """<h1 style='text-align: center;'>Loan Default Predictor</h1>"""
body = ["""The application helps is evaluating a loan applicant's profile based on his application details, and predicts the chance of that applicant defaulting the loan.""",
        "The application also provides a recommendation whether to approve a loan to the applicant or not.",
        "This application is trained on [Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) Dataset.",
        "Null values are treated strategically, Features are selected based on correlation and dropped Multicollinear features, OneHot encoded Categorical features. Upsampled training data using SMOTE for better training.",
        "This app uses a trained Light Gradient Boosting Machine algorithm to provide relaible, accurate and fast predictions."]

st.markdown(title, unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        for text in body:
            st.write(text)
    
    with col2:
        st_lottie(lottie_loan, key='lappy')
    


st.write('---')
text = "<h2 style='text-align: center;'>Data Analysis & Visualizations</h2>"
st.markdown(text, unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image_loan_amt_hist, caption='Histogram of Loan Amount')
    with col2:
        st.image(image_int_rate_hist, caption='Histogram of Interest Rate')
    with col3:
        st.image(image_correlation, caption='Correlation of features')

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image_violinplot_loanamt, caption='Violin Plot of Loan Amount')
    with col2:
        st.image(image_corr_encoding, caption='Correlation of Encoded Features with Target')
    with col3:
        st.image(image_smote, caption='Upsampled Training data with SMOTE')

with st.container():
    st.write("---")
    text = """<h3 style= 'text-align: center;'> Made by Praveen Samudrala </h3>"""
    st.markdown(text, unsafe_allow_html=True)

# Session State
# if 'key' not in st.session_state:
#     st.session_state['key'] = 'value'