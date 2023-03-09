import streamlit as st
import numpy as np
import sklearn
import pandas as pd
pd.set_option('display.max_columns', 50)
import pickle
ohe, lightGBM = pickle.load(open('encoder-model.pkl', 'rb'))


def predict(data):  
  cols = ['loan_amnt', 'term', 'int_rate', 'sub_grade', 'home_ownership',
            'annual_inc', 'verification_status', 'purpose', 'dti', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'initial_list_status', 'application_type', 'mort_acc',
            'pub_rec_bankruptcies', 'earliest_cr_year', 'zip_code']
  cat_col = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'zip_code']

  new_cust = pd.DataFrame(data, columns = cols)
  st.dataframe(new_cust)

  new_cust_onehot = pd.DataFrame(ohe.transform(new_cust[cat_col]))
  new_cust = pd.concat([new_cust.drop(cat_col, axis=1), new_cust_onehot], axis=1)

  pred = lightGBM.predict(new_cust.values)
  pred_probs = lightGBM.predict_proba(new_cust.values)
  print(pred_probs)

  if pred==1: pred_text = """<h2 style="color:white;text-align:center;">Customer is not likely to pay back the loan. Loan can't be approved</h2>"""
  else: pred_text =  """<h2 style="color:white;text-align:center;">Customer is likely to pay back the loan. Loan can be approved</h2>"""

  return pred_text

def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Default Predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write('##')

    #form = st.form(key='Inputs')
    col1, col2 = st.columns(2)
    with col1:
        loan_amt = st.number_input('Loan Amount', min_value= 0)
        term = st.selectbox('Number of Months to repay', 
                            ('36', '60'))
        int_rate = st.number_input('Interest Rate in %',min_value=0)
        sub_grade = st.selectbox('LC Assigned Loan SubGrade', 
                                 ('A1','A2','A3','A4','A5',
                                  'B1','B2','B3','B4','B5',
                                  'C1','C2','C3','C4','C5',
                                  'D1','D2','D3','D4','D5',
                                  'E1','E2','E3','E4','E5',
                                  'F1','F2','F3','F4','F5',
                                  'G1','G2','G3','G4','G5'))
        home_ownership = st.selectbox('Home Ownership Status',
                                 ('RENT', 'MORTGAGE', 'OWN', 'OTHER'))
        annual_inc = st.number_input('Annual Income',min_value=10000)
        verification_status = st.selectbox('Income or Income Source Verification Status',
                                 ('Not Verified', 'Source Verified', 'Verified'))
        purpose = st.selectbox('Loan Purpose', 
                                 ('vacation','debt_consolidation','credit_card','medical',
                                  'home_improvement','small_business','major_purchase','wedding',
                                  'car','moving','house','educational','renewable_energy','other'))
        dti = st.number_input('DTI (5 - 20)',min_value=5, max_value=20)
        open_acc = st.number_input('Number of Open Credit Lines (5 - 50)',min_value=5, max_value=90)
        
    with col2:
        pub_rec = st.number_input('Number of Derogatory Public Records (0 - 50)',min_value=0, max_value=50)
        revol_bal = st.number_input('Total Credit Revolving Balance (1000 - 1000000)',min_value=1000, max_value=1000000)
        revol_util = st.number_input('Revolving Line Utilization Rate (10 - 50)',min_value=10, max_value=900)
        total_acc = st.number_input('The Total Number of Credit Lines (0 - 50)',min_value=2, max_value=150)
        initial_list_status = st.selectbox('Listing Status of Loan',('f', 'w'))
        application_type = st.selectbox('Application Type',('INDIVIDUAL', 'JOINT'))
        mort_acc = st.number_input('Number of Mortgage Accounts (0 - 27)',min_value=0, max_value=27)
        pub_rec_bankruptcies = st.number_input('Number of Public Record Bankruptcies (0 - 10)',min_value=0, max_value=10)
        earliest_cr_year = st.number_input('Earliest Credit Opening Month')
        zip_code = st.selectbox('ZIP Code',
                                 ('22690', '05113', '00813', '11650',
                                  '30723', '70466', '29597', '48052',
                                  '86630', '93700'))
    data = [[loan_amt, term, int_rate, sub_grade, home_ownership, annual_inc, verification_status, purpose, dti, open_acc, pub_rec, revol_bal, revol_util, total_acc, initial_list_status, application_type, mort_acc, pub_rec_bankruptcies, earliest_cr_year, zip_code]]
    
    st.write('---')

    with st.container():
        with st.columns(5)[2]:
            flag = st.button('Submit')
    
    with st.container():
        if flag:
            result = predict(data)
            st.markdown(result, unsafe_allow_html=True)
    
    # HTML code and print using markdown.
    # Use CSS to style the input fields.

    # if st.button("Predict"):
    #     pred = predict(input_text)
    #     if pred == 1:
    #         result = "It's a Disaster"
    #     else:
    #         result = "It's NOT a Disaster"
    # st.success(result)
    if st.button("About"):
        st.text("Built by Praveen Samudrala using Streamlit")

if __name__=='__main__':
    main()