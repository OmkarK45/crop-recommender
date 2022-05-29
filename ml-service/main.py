"""
    Crop Recommendation System - Flask API
    This microservice is responsible for the recommendation of crops based on the user's location.
    The data to be feed to this microservice comes from the Node.JS service running on Railway.
"""
# Library Imports
from flask import Flask, jsonify, request
import os
import pickle
import numpy as np
import pandas as pd
import io
import torch
from PIL import Image
from torchvision import transforms
from flask_cors import CORS

# Local Imports
from constants.diseases import disease_classes
from lib.resnet import ResNet9


app = Flask(__name__)
CORS(app)
"""
    Importing the model [Crop Recommendation Model]
    The model is saved as a pickle file.
    The model is loaded using the pickle module.
"""
crop_recommendation_model = pickle.load(open('models/random-forest.pkl', 'rb'))
yield_prediction_model = pickle.load(open('models/yield_new.pkl', 'rb'))
"""
    Importing the model [Disease Prediction Model]
    The model is saved as a pth file.
    ResNet9 is used for the prediction.
"""
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    'models/leaf-disease-model.pth', map_location=torch.device('cpu')))
disease_model.eval()


@app.route('/')
def index():
    port = int(os.environ.get('PORT', 5000))
    return jsonify({
        "Welcome": "Welcome to the Crop Recommendation System",
        "Status": "OK",
        "Version": "1.0",
        "Port": port,
        "Name": "Machine Learning Microservice"
    })


@ app.route("/predict-crop", methods=['POST'])
def predict_crop():
    if request.method == 'POST':
        """
            The request.json is used to get the data from the Node.JS API.
            Shape of the data
            data = {
                nitrogen: 0.0,
                phosphorus: 0.0,
                potassium: 0.0,
                ph : 0.0,
                rainfall: 0.0, [Fetched using openweather api in NodeJS]
                temperature: 0.0, [Fetched using openweather api in NodeJS]
                humidity: 0.0, [Fetched using openweather api in NodeJS]
            }
        """
        data = request.get_json()

        # check of absense of any data
        if data is None:
            return jsonify({"message": "No data found", "status": "Bad Request"})

        nitrogen = data['nitrogen']
        phosphorus = data['phosphorus']
        potassium = data['potassium']
        ph = data['ph']
        rainfall = data['rainfall']
        temperature = data['temperature']
        humidity = data['humidity']

        # Input array to be given to the machine learning model
        input = np.array(
            [[nitrogen, phosphorus, potassium, ph, rainfall, temperature, humidity]])

        # Prediction using the model
        prediction = crop_recommendation_model.predict(input)
        final_prediction = prediction[0]

        # if there is no final prediction
        if final_prediction == 0:
            return jsonify({
                "message": "No crop recommendation found. Consider using other inputs.", "status": "error"
            })

        # Else we send the prediction to the Node.JS API
        return jsonify({
            "prediction": final_prediction,
            "code": "SUCCESS"
        })

    return jsonify({'test': 'test'})

# Route to predict the fertilizer


@app.route('/recommend-fertilizer', methods=['POST'])
def recommend_fertilizer():

    # We take crop and NPK values in the request
    if request.method == 'POST':
        data = request.get_json()

        crop = str(data['crop'])
        nitrogen = int(data['nitrogen'])
        phosphorus = int(data['phosphorus'])
        potassium = int(data['potassium'])

        # We read known fertilizer data from the file
        fertilizer_data = pd.read_csv('data/fertilizer_data.csv')

        # We find amount of nitrogen and phosphorus required for that crop
        nitrogen_required = fertilizer_data[fertilizer_data['Crop']
                                            == crop]['N'].iloc[0]
        phosphorus_required = fertilizer_data[fertilizer_data['Crop']
                                              == crop]['P'].iloc[0]
        potassium_required = fertilizer_data[fertilizer_data['Crop']
                                             == crop]['K'].iloc[0]

        print(nitrogen_required, phosphorus_required, potassium_required)

        # We calculate the amount of fertilizer required
        nitrogen_difference = nitrogen_required - nitrogen
        phosphorus_difference = phosphorus_required - phosphorus
        potassium_difference = potassium_required - potassium

        temp = {
            abs(nitrogen_difference): 'N',
            abs(phosphorus_difference): 'P',
            abs(potassium_difference): 'K'
        }

        # We find the maximum value
        max_value = temp[max(temp.keys())]

        # We run a switch case to determine of one of these is low or high
        if max_value == 'N':
            if nitrogen_difference < 0:
                fertilizer = 'HIGH_NITROGEN'
            else:
                fertilizer = 'LOW_NITROGEN'
        elif max_value == 'P':
            if phosphorus_difference < 0:
                fertilizer = 'HIGH_PHOSPHORUS'
            else:
                fertilizer = 'LOW_PHOSPHORUS'
        else:
            if potassium_difference < 0:
                fertilizer = 'HIGH_POTASSIUM'
            else:
                fertilizer = 'LOW_POTASSIUM'

        return jsonify({"status": "success", "fertilizer": fertilizer})


@app.route('/identify-disease', methods=['POST', "GET"])
def identify_disease():
    if request.method == "POST":
        print(request.files.get('file'))
        if "file" not in request.files:
            return jsonify({"message": "No image upload found. Please double check if you have correctly uploaded file.", "status": "BAD_REQUEST"})

        input_image_file = request.files['file']
        if not input_image_file:
            return jsonify({"message": "No image upload found. Please double check if you have correctly uploaded file.", "status": "BAD_REQUEST"})
        try:
            image = input_image_file.read()

            prediction = predict_image(image)

            return jsonify({"status": "SUCCESS", "prediction": prediction})
        except Exception as e:
            print("ERROR", e)
            return jsonify({"message": "Error in processing the image. Please try again.", "status": "INTERNAL_SERVER_ERROR"})


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


@app.route("/predict-yield", methods=['POST'])
def predict_yield():
    if request.method == 'POST':
        data = request.get_json()
        state_name = data['state_name']
        crop = data['crop']
        area = data['area']
        soil_type = data['soil_type']
        print(state_name, crop, area, soil_type)

        pred_args = [state_name, crop, area, soil_type]
        pred_args_arr = np.array(pred_args)
        pred_args_arr = pred_args_arr.reshape(1, -1)
        output = yield_prediction_model.predict(pred_args_arr)
        print(output)
        pred = format(int(output[0]))
        Yield = int(pred) / float(area)
        yields = Yield*1000
        prediction_text = pred
        return jsonify({"status": "success", "yield": yields, "production": prediction_text})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=os.getenv("PORT", default=5000))
