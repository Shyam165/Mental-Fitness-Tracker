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
        schizophrenia = request.form['schizophrenia']
        bipolar = request.form['bipolar']
        eating = request.form['eating']
        anxiety = request.form['anxiety']
        drug_use = request.form['drug_use']
        depressive = request.form['depressive']
        alcohol_use = request.form['alcohol_use']

        # Check if any input field is empty
        if '' in [schizophrenia, bipolar, eating, anxiety, drug_use, depressive, alcohol_use]:
            return render_template('index.html', error_message='Please fill all input fields.')

        # Convert input values to float
        schizophrenia = float(schizophrenia)
        bipolar = float(bipolar)
        eating = float(eating)
        anxiety = float(anxiety)
        drug_use = float(drug_use)
        depressive = float(depressive)
        alcohol_use = float(alcohol_use)

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

        # Interpret the prediction
        interpretation = "Based on the provided information, it is estimated that approximately {}% of the population is affected by mental disorders.".format(round(predicted_dalys, 4))

        # Render prediction template with the result
        return render_template('prediction.html', predicted_dalys=round(predicted_dalys, 4), interpretation=interpretation)

if __name__ == '__main__':
    app.run(debug=True)
