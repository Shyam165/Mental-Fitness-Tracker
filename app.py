from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        schizophrenia = float(request.form['schizophrenia'])
        bipolar = float(request.form['bipolar'])
        eating = float(request.form['eating'])
        anxiety = float(request.form['anxiety'])
        drug_use = float(request.form['drug_use'])
        depressive = float(request.form['depressive'])
        alcohol_use = float(request.form['alcohol_use'])

        # Prepare user input as DataFrame for prediction
        user_input = pd.DataFrame({
            'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': [schizophrenia],
            'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': [bipolar],
            'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': [eating],
            'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': [anxiety],
            'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)': [drug_use],
            'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': [depressive],
            'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)': [alcohol_use]
        })

        # Make prediction
        predicted_dalys = model.predict(user_input)[0]

        # Render prediction template with the result
        return render_template('prediction.html', predicted_dalys=round(predicted_dalys, 4))

if __name__ == '__main__':
    app.run(debug=True)
