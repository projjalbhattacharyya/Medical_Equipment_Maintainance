from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Load models
models = {
    'mri': joblib.load('models/mri.pkl'),
    'bloodTest': joblib.load('models/blood_test_model.pkl'),
    'ventilator': joblib.load('models/ventilater.pkl'),
    'ct_scan': joblib.load('models/ctScan.pkl'),
    'ct_scan_fail': joblib.load('models/CT_Scan_error.pkl'),
    'ct_scan_error': joblib.load('models/CT_Scan_fail.pkl')
}

scaler = StandardScaler()

# Preprocessing function of mri
def preprocess_new_data_mri(new_data_mri):
    # Load the MRI scaler
    scaler_mri = joblib.load('models/scaler_mri.pkl')  # Adjust path as necessary

    # Convert date columns to numerical values (Unix timestamp)
    date_columns = [col for col in new_data_mri.columns if 'Date' in col]
    for col in date_columns:
        new_data_mri[col] = pd.to_datetime(new_data_mri[col], errors='coerce').astype('int64') // 10**9

    # Ensure all features are numeric
    new_data_mri = new_data_mri.select_dtypes(include=[np.number])

    # Feature scaling using the loaded scaler
    new_data_mri = scaler_mri.transform(new_data_mri)

    return new_data_mri

# Preprocessing function of ct scan
def preprocess_new_data_ctscan(new_data_ctscan):
    # 1. Years Since Installation
    new_data_ctscan['Years Since Installation'] = new_data_ctscan['Years Since Installation']

    # 2. Days Since Last Maintenance
    new_data_ctscan['Days Since Last Maintenance'] = new_data_ctscan['Days Since Last Maintenance']

    # 3. Error Rate (Error Logs Count per Usage Hour)
    new_data_ctscan['Error Rate'] = new_data_ctscan['Error Logs Count'] / new_data_ctscan['Usage Hours']
    new_data_ctscan['Error Rate'] = new_data_ctscan['Error Rate'].fillna(0)

    # 4. Temperature Deviation
    mean_temp = new_data_ctscan['Temperature (°C)'].mean()
    new_data_ctscan['Temperature Deviation'] = new_data_ctscan['Temperature (°C)'] - mean_temp

    # 5. Maintenance Gap Ratio
    new_data_ctscan['Maintenance Gap Ratio'] = new_data_ctscan['Days Since Last Maintenance'] / (new_data_ctscan['Years Since Installation'] * 365)
    new_data_ctscan['Maintenance Gap Ratio'] = new_data_ctscan['Maintenance Gap Ratio'].fillna(0)

    # 6. High Error Frequency Indicator
    new_data_ctscan['High Error Frequency'] = (new_data_ctscan['Error Logs Count'] > new_data_ctscan['Error Logs Count'].mean()).astype(int)

    # 7. Device Efficiency
    new_data_ctscan['Device Efficiency'] = new_data_ctscan['Detector Sensitivity (%)'] / new_data_ctscan['Radiation Dosage (mGy)']
    new_data_ctscan['Device Efficiency'] = new_data_ctscan['Device Efficiency'].fillna(0)

    # 8. Time Since Maintenance Percentage
    new_data_ctscan['Maintenance Time %'] = (new_data_ctscan['Days Since Last Maintenance'] / (new_data_ctscan['Years Since Installation'] * 365)) * 100
    new_data_ctscan['Maintenance Time %'] = new_data_ctscan['Maintenance Time %'].fillna(0)

    # Display the updated new_dataFrame with new features
    new_features_ctscan = new_data_ctscan[[
        'Usage Hours',
    'Temperature (°C)',
    'Error Logs Count',
    'Detector Sensitivity (%)',
    'Radiation Dosage (mGy)',
    'Scan Count',
    'Humidity (%)',
    'Days Since Last Maintenance',
    "Years Since Installation"
    ]]

    return new_features_ctscan

def preprocess_new_data_ventilater(new_data_ventilater):
    scaler_ventilater = joblib.load('models/scaler_ventilater.pkl')
    # Define the predictors as used during model training
    predictors = [
        'Usage Hours', 'Temperature (°C)', 'Humidity (%)',
        'Oxygen Delivery Rate (L/min)', 'Pressure Settings (cmH2O)',
        'Patient Usage Count', 'Last Maintenance Ordinal'
    ]
    
    # Ensure that the new data contains all required columns
    missing_columns = [col for col in predictors if col not in new_data_ventilater.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle missing values in the new data
    for col in predictors:
        if new_data_ventilater[col].isnull().sum() > 0:
            if new_data_ventilater[col].dtype in ['float64', 'int64']:
                new_data_ventilater[col].fillna(new_data_ventilater[col].median(), inplace=True)
            else:
                new_data_ventilater[col].fillna(new_data_ventilater[col].mode()[0], inplace=True)
    
    # Reorder the columns to match the training data
    new_data_ventilater = new_data_ventilater[predictors]
    
    # Standardize numerical features using the loaded scaler
    scaled_data = scaler_ventilater.transform(new_data_ventilater)
    
    # Return the preprocessed data as a DataFrame
    return pd.DataFrame(scaled_data, columns=predictors)

# Define output mapping for all models
output_classes = {0: 'Not Failure', 1: 'Failure'}
output_classes_error = {0: 'Image Distortion', 1: 'Radiation Overdose',2:'Detector Malfunction'}

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('homepage.html')

@app.route('/mri', methods=['GET', 'POST'])
def mri():
    if request.method == 'POST':
        try:
            # Get data from the form
            form_data = {key: request.form[key] for key in request.form}
            form_data = {k: float(v) for k, v in form_data.items()}
            new_data_mri = pd.DataFrame([form_data])

            # Preprocessing
            transformed_features = preprocess_new_data_mri(new_data_mri)

            # Make prediction
            prediction = models['mri'].predict(transformed_features)[0]
            result = output_classes[prediction]

            return render_template('mri.html', result=result)
        except Exception as e:
            return render_template('mri.html', result=f'Error: {str(e)}')

    return render_template('mri.html', result=None)


@app.route('/ctScan', methods=['GET', 'POST'])
def ct_scan():
    if request.method == 'POST':
        try:
            # Get data from the form
            form_data = {key: float(request.form[key]) for key in request.form}
            new_data_ctscan = pd.DataFrame([form_data])

            # Preprocess the input
            transformed_features = preprocess_new_data_ctscan(new_data_ctscan)

            # First prediction
            prediction1 = models['ct_scan_fail'].predict(transformed_features)[0]
            result1 = output_classes[prediction1]

            # Initialize result2 with None
            result2 = None

            # If failure, proceed with the second model
            if result1 == 'Failure':
                prediction2 = models['ct_scan_error'].predict(transformed_features)[0]
                result2 = output_classes_error[prediction2]

            return render_template('ctScan.html', result1=result1, result2=result2)

        except Exception as e:
            return render_template('ctScan.html', result1=None, result2=f'Error: {str(e)}')

    return render_template('ctScan.html', result1=None, result2=None)

@app.route('/bloodTest', methods=['GET', 'POST'])
def bloodTest():
    if request.method == 'POST':
        try:
            # Get data from the form
            form_data = {
                'Usage Hours': float(request.form['Usage Hours']),
                'Error Logs Count': int(request.form['Error Logs Count']),
                'Temperature (°C)': float(request.form['Temperature (°C)']),
                'Humidity (%)': float(request.form['Humidity (%)']),
                'Test Types Conducted': int(request.form['Test Types Conducted']),
                'Sample Count': int(request.form['Sample Count']),
                'Reagent Levels (%)': float(request.form['Reagent Levels (%)'])
            }

            # Convert form data to a DataFrame
            input_data = pd.DataFrame([form_data])

            # Preprocess the input
            predictors = [
                'Usage Hours', 'Error Logs Count', 'Temperature (°C)',
                'Humidity (%)', 'Test Types Conducted', 'Sample Count', 'Reagent Levels (%)'
            ]

            # Ensure input columns are aligned with predictors
            input_data = input_data[predictors]

            # Make the prediction
            prediction = models['bloodTest'].predict(input_data)[0]
            result = output_classes[prediction]

            return render_template('bloodTest.html', result=result)
        except Exception as e:
            return render_template('bloodTest.html', result=f'Error: {str(e)}')

    return render_template('bloodTest.html', result=None)


@app.route('/ventilater', methods=['GET', 'POST'])
def ventilater():
    if request.method == 'POST':
        try:
            # Extract data from the form
            input_data = {
                'Last Maintenance Ordinal': [float(request.form['Last Maintenance Ordinal'])],
                'Usage Hours': [float(request.form['Usage Hours'])],
                'Temperature (°C)': [float(request.form['Temperature (°C)'])],
                'Humidity (%)': [float(request.form['Humidity (%)'])],
                'Oxygen Delivery Rate (L/min)': [float(request.form['Oxygen Delivery Rate (L/min)'])],
                'Pressure Settings (cmH2O)': [float(request.form['Pressure Settings (cmH2O)'])],
                'Patient Usage Count': [float(request.form['Patient Usage Count'])]
            }
            new_data = pd.DataFrame(input_data)

            # Preprocess the input
            transformed_features = preprocess_new_data_ventilater(new_data)

            # Make a prediction
            prediction = models['ventilator'].predict(transformed_features)[0]
            result = output_classes[prediction]
            
            return render_template('ventilater.html', result=result)
        except Exception as e:
            return render_template('ventilater.html', result=f'Error: {str(e)}')

    return render_template('ventilater.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)