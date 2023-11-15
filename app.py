#flask 
#pickle
#numpy

# from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask import *
import pandas as pd
from model.model1 import train_model1
from model.model2 import train_model2


app = Flask(__name__)

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret_key1'

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

        # Load the trained models
        with open('output/model1_predictions.pkl', 'rb') as model_file1:
            model1 = pickle.load(model_file1)

        with open('output/model2_predictions.pkl', 'rb') as model_file2:
            model2 = pickle.load(model_file2)


        # Perform weighted average computation
        weight1 = 0.7  # Adjust weights as needed
        weight2 = 0.3

        output = (weight1 * model1[inputValue-1]) + (weight2 * model2[inputValue-1])

        # Return the prediction as JSON
        return jsonify({'prediction': float(output[0])})

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        print(f)

        data_filename = secure_filename(f.filename)
 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                            data_filename))
 
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],
                        data_filename)
        uploaded_df = pd.read_csv(session['uploaded_data_file_path'],
                              encoding='unicode_escape')
        # Converting to html Table
        uploaded_df_html = uploaded_df.to_html()

        return render_template('index.html',data_var=uploaded_df_html)
 

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        if session['uploaded_data_file_path'] == None:
            raise Exception("No file path")
        
        file_path = session['uploaded_data_file_path']

        print(os.getcwd())

        train_model1(file_path)
        train_model2(file_path)

        return render_template('index.html')
        

 



if __name__ == '__main__':
    app.run(debug=True)
