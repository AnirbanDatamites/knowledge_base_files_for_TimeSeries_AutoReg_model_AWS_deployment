
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd


# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input from the form
        steps = int(request.form['steps'])
        
        # Load the dataset to ensure we use the same data structure
        #data = pd.read_csv('Electric Production_Time Series.csv', parse_dates=[0], index_col='DATE')
        #data.rename(columns={'IPG2211A2N': 'Electricity_Production'}, inplace=True)
        
        # Forecast the specified number of steps
        prediction = model.forecast(steps=steps)
        
        # Prepare results for rendering
        prediction = np.round(prediction, 2).tolist()
        return render_template('index.html', prediction_text=f'Predicted values for next {steps} steps: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)

#Note: please check your file path of your model and modify the model path over here accordigly and also you can change the port at the end based on your requirement, instead of port=8080, you can use any other port to run your model based on your situation.     


