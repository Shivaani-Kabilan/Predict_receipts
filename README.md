# Predict_receipts

Build the docker image and run the container

docker build . -t receipts_predictor:latest
docker run -p 3000:5000 -td receipts_predictor


check if container is running: docker ps

Check 127.0.0.1:3000 address for the UI
