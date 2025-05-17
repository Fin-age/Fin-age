import pandas as pd
from flask import Flask, render_template, request


import xgboost as xgb



# Load the model and scaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# Load dataset
data = pd.read_csv('loan_approval_dataset.csv')
app = Flask(__name__)
# Basic cleaning
data.columns = data.columns.str.strip()  # Strip column names
loan_status_counts = data['loan_status'].value_counts()
print("Loan Status Counts:\n", loan_status_counts)

# Label encoding for all categorical features
label_encoder = LabelEncoder()
categorical_columns = ['education', 'self_employed', 'loan_status']  # List of categorical columns
for col in categorical_columns:
    if col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])
loan_status_counts = data['loan_status'].value_counts()
print(loan_status_counts)

# Add new features
data['loan_to_income_ratio'] = data['loan_amount'] / data['income_annum'].replace(0, 1)  # Avoid division by zero
data['cibil_to_loan_ratio'] = data['cibil_score'] / data['loan_amount'].replace(0, 1)  # Avoid division by zero

# Additional features
data['loan_to_cibil_ratio'] = data['loan_amount'] / data['cibil_score'].replace(0, 1)  # Avoid division by zero
data['loan_term_to_income_ratio'] = data['loan_term'] / data['income_annum'].replace(0, 1)  # Avoid division by zero
data['luxury_to_income_ratio'] = data['luxury_assets_value'] / data['income_annum'].replace(0, 1)  # Avoid division by zero
data['total_assets_to_loan_ratio'] = (data['residential_assets_value'] + 
                                        data['commercial_assets_value'] + 
                                        data['luxury_assets_value']) / data['loan_amount'].replace(0, 1)  # Avoid division by zero

# Selecting top features based on your analysis
top_features = ['cibil_score', 'income_annum', 'loan_amount', 'loan_term', 'luxury_assets_value', 
                'loan_to_income_ratio', 'cibil_to_loan_ratio', 'loan_to_cibil_ratio', 
                'loan_term_to_income_ratio', 'luxury_to_income_ratio', 'total_assets_to_loan_ratio']

# Features and target
X = data[top_features]
y = data['loan_status']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_resampled, y_resampled)

# Classification report
y_pred = xgb_model.predict(X_test)


# Dump the model
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to predict loan status
def predict_loan_status(features):
    # Load the model and scaler
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Create a DataFrame for the input features
    input_df = pd.DataFrame([features], columns=['cibil_score', 'income_annum', 'loan_amount', 'loan_term', 'luxury_assets_value'])
    
    # High CIBIL and Low Income thresholds
    high_cibil_threshold = 750  # Define your high CIBIL threshold
    low_income_threshold = 300000  # Define your low income threshold

    # Check for high CIBIL and low income
    if input_df['cibil_score'][0] > high_cibil_threshold and input_df['income_annum'][0] < low_income_threshold:
        return "Rejected due to high CIBIL score and low income."

    # Generate interaction features for the input
    input_df['loan_to_income_ratio'] = input_df['loan_amount'] / input_df['income_annum'].replace(0, 1)  # Avoid division by zero
    input_df['cibil_to_loan_ratio'] = input_df['cibil_score'] / input_df['loan_amount'].replace(0, 1)  # Avoid division by zero
    input_df['loan_to_cibil_ratio'] = input_df['loan_amount'] / input_df['cibil_score'].replace(0, 1)  # Avoid division by zero
    input_df['loan_term_to_income_ratio'] = input_df['loan_term'] / input_df['income_annum'].replace(0, 1)  # Avoid division by zero
    input_df['luxury_to_income_ratio'] = input_df['luxury_assets_value'] / input_df['income_annum'].replace(0, 1)  # Avoid division by zero
    input_df['total_assets_to_loan_ratio'] = (input_df['luxury_assets_value']) / input_df['loan_amount'].replace(0, 1)

    # Ensure the input features are in the same order as training
    input_df = input_df[top_features]
    
    # Scale the input features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return "Accepted" if prediction[0] == 0 else "Rejected"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Get user input from the form
            cibil_score = float(request.form['cibil_score'])
            income_annum = float(request.form['income_annum'])
            loan_amount = float(request.form['loan_amount'])
            loan_term = float(request.form['loan_term'])
            luxury_assets_value = float(request.form['luxury_assets_value'])
        except ValueError:
            result = "Please enter valid numerical values."
            return render_template('index.html', result=result)

        features = [cibil_score, income_annum, loan_amount, loan_term, luxury_assets_value]
        
        # Predict loan status
        result = predict_loan_status(features)
    
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
