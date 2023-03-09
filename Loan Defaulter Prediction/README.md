<h1 align="center">Loan Defaulter Predictor App</h1>

# Overview: 

The Loan Defaulter Predictor Application analyses a loan applicant's data and predicts the likeliness of whether the applicant will default the loan or not based on Lending Club's data. The application provides a capability to use a LightGBM model and TF based Neural Network(in works) for prediction, providing choice of model to use.

# Project Planning:

## 1.  Data Exploration
  * Checking missing data using heatmaps.
  * Distribution of dependent variable
  * Visualization of different features to understand relations and distributions.
  * Understanding the correlation between features using heatmaps.

## 2. Data Preprocessing  
  * Handling Null Values - Strategic Imputation with corerlated features and Median.
  * Feature Selection based on Correlation with target and to tackle Multicollinearity.
  * Feature Encoding - Encoded Categorical features and Normalized numerical features.
  * Splitting of Training and Testing data.
  * Upsampling Training data with SMOTE to tackle imbalanced classes.

## 3. Model Building and Evaluation
  * Model Experimentation
    - Logistic Regression
    - Random Forest
    - LightGBM
    - CatBoost
    - XGBosst
    - Sequential ANN model.

  * Evaluation
    - Predictions on test set
    - Evaluation of model with classification report.
    - Evaluating Performance Metrics - TP,TN,FP,FN,Precision,Recall,F1-score,AUROC,AUPRC.

<img
  src="https://github.com/Praveen-Samudrala/Machine-Learning-and-Data-Science/blob/main/Loan%20Defaulter%20Prediction/images/performance.png"
  alt="performance"
  title="performance"/>

<img
  src="https://github.com/Praveen-Samudrala/Machine-Learning-and-Data-Science/blob/main/Loan%20Defaulter%20Prediction/images/performance2.png"
  alt="performance2"
  title="performance2"/>

## 4. Inference function
  * Training a model on whole dataset (X and y) without splitting.
  * Function "assess_customer" to assess new customers whether they payback loan or not.
  * Check model predictions on new customer.

## 5. Prediction App with Streamlit
<img
  src="https://github.com/Praveen-Samudrala/Machine-Learning-and-Data-Science/blob/main/Loan%20Defaulter%20Prediction/images/screen1.png"
  alt="Home Page_1"
  title="Home Page"/>

<img
  src="https://github.com/Praveen-Samudrala/Machine-Learning-and-Data-Science/blob/main/Loan%20Defaulter%20Prediction/images/screen2.png"
  alt="Home Page_2"
  title="Home Page"/>

<img
  src="https://github.com/Praveen-Samudrala/Machine-Learning-and-Data-Science/blob/main/Loan%20Defaulter%20Prediction/images/screen3.png"
  alt="Predictor_1"
  title="Predictor"/>

<img
  src="https://github.com/Praveen-Samudrala/Machine-Learning-and-Data-Science/blob/main/Loan%20Defaulter%20Prediction/images/screen4.png"
  alt="Predictor_2"
  title="Predictor"/>
  
## Launch
Install the dependencies via: 
```
pip install -r requirements.txt
```

Run the application using:
(Type 1 and `Tab` to autocomplete app name)
```
streamlit run 1_🏠_home.py
```