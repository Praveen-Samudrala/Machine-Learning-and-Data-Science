import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn import linear_model
import pandas as pd
pd.set_option('display.max_columns', 50)
import pickle

app = Flask(__name__)
ohe, scaler, log_reg = pickle.load(open('onehot-scaler-logistic_objects.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
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


def access_customer_new(new_cust):
  cat_col = new_cust.select_dtypes(exclude=[np.number]).columns.values
  dummy = pd.get_dummies(new_cust[cat_col], drop_first=True)
  new_cust = pd.concat([new_cust,dummy],axis=1).drop(cat_col,axis=1)

  #print(f'new_cust 3 rows shape:{new_cust.shape}')
  new_cust = new_cust.reset_index(drop=True)

  df1 = pd.read_csv('/content/df1.csv', index_col=0).drop('loan_repaid', axis=1)
  #print(f'DF1 shape:{df1.shape}')

  df1, new_cust = df1.align(new_cust, axis=1, fill_value=0)

  trainCols = ['annual_inc', 'application_type_INDIVIDUAL',
        'application_type_JOINT', 'dti', 'earliest_cr_year',
        'home_ownership_OTHER', 'home_ownership_OWN',
        'home_ownership_RENT', 'initial_list_status_w', 'int_rate',
        'loan_amnt', 'mort_acc', 'open_acc', 'pub_rec',
        'pub_rec_bankruptcies', 'purpose_credit_card',
        'purpose_debt_consolidation', 'purpose_educational',
        'purpose_home_improvement', 'purpose_house',
        'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
        'purpose_other', 'purpose_renewable_energy',
        'purpose_small_business', 'purpose_vacation', 'purpose_wedding',
        'revol_bal', 'revol_util', 'sub_grade_A2', 'sub_grade_A3',
        'sub_grade_A4', 'sub_grade_A5', 'sub_grade_B1', 'sub_grade_B2',
        'sub_grade_B3', 'sub_grade_B4', 'sub_grade_B5', 'sub_grade_C1',
        'sub_grade_C2', 'sub_grade_C3', 'sub_grade_C4', 'sub_grade_C5',
        'sub_grade_D1', 'sub_grade_D2', 'sub_grade_D3', 'sub_grade_D4',
        'sub_grade_D5', 'sub_grade_E1', 'sub_grade_E2', 'sub_grade_E3',
        'sub_grade_E4', 'sub_grade_E5', 'sub_grade_F1', 'sub_grade_F2',
        'sub_grade_F3', 'sub_grade_F4', 'sub_grade_F5', 'sub_grade_G1',
        'sub_grade_G2', 'sub_grade_G3', 'sub_grade_G4', 'sub_grade_G5',
        'term', 'total_acc', 'verification_status_Source Verified',
        'verification_status_Verified', 'zip_code_05113', 'zip_code_11650',
        'zip_code_22690', 'zip_code_29597', 'zip_code_30723',
        'zip_code_48052', 'zip_code_70466', 'zip_code_86630',
        'zip_code_93700']
  new_cust = new_cust[trainCols]

  cust = new_cust.values.reshape(1,77)
  cust_sca = scaler.transform(cust)

  #y_pred_ann = (final_model.predict(X_test_sca) > 0.5).astype("int32")
  #print(y_pred_ann)
  pred = final_model.predict(cust_sca)
  #print(f'Prediction: {pred[0]}')
  if pred>=0.5:
    print ("Customer is likely to pay back the loan \nLoan can be approved")
  else:
    print ("Customer is not likely to pay back the loan \nLoan can't be approved")

if __name__ == "__main__":
    app.run(debug=True)