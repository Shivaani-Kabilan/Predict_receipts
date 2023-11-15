#flask 
#pickle
#numpy

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models
with open('model1_predictions.pkl', 'rb') as model_file1:
    model1 = pickle.load(model_file1)

with open('model2_predictions.pkl', 'rb') as model_file2:
    model2 = pickle.load(model_file2)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the request
        data = request.get_json()
        inputValue = int(data['inputValue'])

        # Perform weighted average computation
        weight1 = 0.7  # Adjust weights as needed
        weight2 = 0.3

        output = (weight1 * model1[inputValue-1]) + (weight2 * model2[inputValue-1])

        # Return the prediction as JSON
        return jsonify({'prediction': float(output[0])})

if __name__ == '__main__':
    app.run(debug=True)
