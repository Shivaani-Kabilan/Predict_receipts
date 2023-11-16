# Predict_receipts

This is an application to predict the monthly count of scanned receipts based on the previous year's daily count of scanned receipts.


## How to Run

Build the docker image and run the container.

To build and run the docker container:

docker build . -t receipts_predictor:latest

docker run -p 3000:5000 -td receipts_predictor


To check if container is running, use: docker ps

Check 127.0.0.1:3000 address for the UI


## Machine Learning Model
Machine Learning Framework used: TensorFlow

To make predicitons, I have built 2 ML models and then taken a weighted sum of them.

Model 1: Predicts the approximate number of scanned receipts for each month of 2022, based on number of scanned receipts for the previous month

Model 2: Predicts the approximate number of scanned receipts for each month of 2022, based on the difference between the number of scanned receipts of consecutive months

On a high level, the ML model architecture involves simple neural network (Sequential model) created using TensorFlow's Keras API. 

## Web application
Web framework used for building the app: Flask

The web app has the following features:

Predict: When the user enters a month value (1 to 12), the predicted number of scanned receipts of that month in the year 2022 will be displayed.

Choose file: This function enables the user to enter any dataset of their choice (similar to the structure of the given dataset "data_daily.csv"); this helps in dynamially training the model, as and when more data becomes available.

Upload: It allows the user to upload the dataset in .csv format, and then a preview of the uploaded dataset will be displayed.

Train: Click on this button to train the ML models.

Visualize: It visualizes the data and gives a plot, showing how the monthly scanned receipts has changed over the months in the years 2021 and 2022.


