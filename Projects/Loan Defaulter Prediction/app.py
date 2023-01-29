import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
pd.set_option('display.max_columns', 50)
import pickle
import tensorflow as tf

app = Flask(__name__)
scaler = pickle.load(open('scaler.pkl', 'rb'))
my_model = tf.keras.models.load_model('my_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

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
  cat_col = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type', 'zip_code']

  new_cust = pd.DataFrame(features, columns = cols)
  print(new_cust)

  dummy = pd.get_dummies(new_cust[cat_col], drop_first=True)
  new_cust = pd.concat([new_cust,dummy],axis=1).drop(cat_col,axis=1)
  new_cust = new_cust.reset_index(drop=True)

  df1 = pd.read_csv('/content/df1.csv', index_col=0).drop('loan_repaid', axis=1)
  df1, new_cust = df1.align(new_cust, axis=1, fill_value=0)
  df1_cols = df1.columns.values
  
  new_cust = new_cust[df1_cols]
  new_cust = new_cust.values.reshape(1,77)

  new_cust_sca = scaler.transform(new_cust)

  pred = my_model.predict_classes(new_cust_sca)

  if pred == 1: output = "Customer is likely to pay back the loan \nLoan can be approved"
  else: output = "Customer is not likely to pay back the loan \nLoan can't be approved"

  return render_template('index.html', prediction_text=f'{output}')


if __name__ == "__main__":
    app.run(debug=True)