import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn import linear_model
import pandas as pd
pd.set_option('display.max_columns', 50)
import pickle

app = Flask(__name__)
ohe, lightGBM = pickle.load(open('encoder-model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
  features = np.array([[int(request.form['loan_amount']), int(request.form['term']), int(request.form['int_rate']), request.form['sub_grade'],
                        request.form['home_ownership'], int(request.form['annual_inc']), request.form['verification_status'], request.form['purpose'], 
                        int(request.form['dti']), int(request.form['open_acc']), int(request.form['pub_rec']), int(request.form['revol_bal']),
                        int(request.form['revol_util']), int(request.form['total_acc']), request.form['initial_list_status'], request.form['application_type'], 
                        int(request.form['mort_acc']), int(request.form['pub_rec_bankruptcies']), int(request.form['earliest_cr_year']), request.form['zip_code']
                        ]])
  
  cols = ['loan_amnt', 'term', 'int_rate', 'sub_grade', 'home_ownership',
            'annual_inc', 'verification_status', 'purpose', 'dti', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'initial_list_status', 'application_type', 'mort_acc',
            'pub_rec_bankruptcies', 'earliest_cr_year', 'zip_code']
  cat_col = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'zip_code']

  new_cust = pd.DataFrame(features, columns = cols)
  print(new_cust)

  new_cust_onehot = pd.DataFrame(ohe.transform(new_cust[cat_col]))
  new_cust = pd.concat([new_cust.drop(cat_col, axis=1), new_cust_onehot], axis=1)

  pred = lightGBM.predict(new_cust.values)

  if pred==1: pred_text = "Customer is not likely to pay back the loan. Loan can't be approved"
  else: pred_text =  "Customer is likely to pay back the loan. Loan can be approved"

  return render_template('predictor.html', prediction_text=f'{pred_text}')

def predict_old():
    '''
    For rendering results on HTML GUI
    '''
    features = np.array([[request.form['loan_amount'], request.form['term'], request.form['int_rate'], request.form['sub_grade'],
                        request.form['home_ownership'], request.form['annual_inc'], request.form['verification_status'], request.form['purpose'], 
                        request.form['dti'], request.form['open_acc'], request.form['pub_rec'], request.form['revol_bal'],
                        request.form['revol_util'], request.form['total_acc'], request.form['initial_list_status'], request.form['application_type'], 
                        request.form['mort_acc'], request.form['pub_rec_bankruptcies'], request.form['earliest_cr_year'], request.form['zip_code']
                        ]])
    cols = ['loan_amnt', 'term', 'int_rate', 'sub_grade', 'home_ownership',
            'annual_inc', 'verification_status', 'purpose', 'dti', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'initial_list_status', 'application_type', 'mort_acc',
            'pub_rec_bankruptcies', 'earliest_cr_year', 'zip_code']
    cat_cols = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'zip_code']

    test_df = pd.DataFrame(features, columns = cols)
    print(test_df)

    new_cust_onehot = pd.DataFrame(ohe.transform(test_df[cat_cols]))
    new_cust = test_df.drop(cat_cols, axis=1)
    new_cust_x = pd.concat([new_cust, new_cust_onehot], axis=1)

    x_target_sca = scaler.transform(new_cust_x)
    pred = log_reg.predict(x_target_sca)
    print(pred)

    if pred == 1: output = "Customer is likely to pay back the loan \nLoan can be approved"
    else: output = "Customer is not likely to pay back the loan \nLoan can't be approved"

    return render_template('index.html', prediction_text=f'{output}')


if __name__ == "__main__":
    app.run(debug=True)