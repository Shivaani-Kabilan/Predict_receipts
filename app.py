#flask 
#pickle
#numpy
#matplotlib

# from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask import *
import pandas as pd
from model.model1 import train_model1
from model.model2 import train_model2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib


app = Flask(__name__)

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv', 'png'}
 
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
        print(secure_filename(f.filename))

        if secure_filename(f.filename) == '': 
            return render_template('index.html')

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



@app.route('/visualize', methods=['POST'])
def visualize():
    if request.method == 'POST':
        months = [ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        # Load the trained models
        with open('output/model1_predictions.pkl', 'rb') as model_file1:
            model1 = pickle.load(model_file1)

        with open('output/model2_predictions.pkl', 'rb') as model_file2:
            model2 = pickle.load(model_file2)

        receipts_2022, receipts_2021 = [], []

        for i in range(12):
            op = (0.7 * model1[i]) + (0.3 * model2[i])
            receipts_2022.append(float(op[0]))

        with open('output/2021receipts.pkl', 'rb') as g:
            receipts_2021 = pickle.load(g)


        # Plotting the data
        matplotlib.pyplot.switch_backend('Agg') 

        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        plt.plot(months, receipts_2021, marker='o', label='2021')
        plt.plot(months, receipts_2022, marker='o', label='2022')

        # Adding labels and title
        plt.xlabel('Months')
        plt.ylabel('Number of Receipts')
        plt.title('Number of Receipts Obtained in 2021 and 2022')
        plt.legend()  # Show legend

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        
        plt.savefig('static/output_graph.png', format='png')

        # Close plot to prevent displaying it directly
        plt.close()

        image_path = 'static/output_graph.png'
        return render_template('show_image.html', image_path=image_path)
        


@app.route('/model_metrics', methods=['POST'])
def model_metrics():
    if request.method == 'POST':
        #image_path = 'static/metrics.png'
        return render_template('show_metrics.html')

 

if __name__ == '__main__':
    app.run(debug=True)
