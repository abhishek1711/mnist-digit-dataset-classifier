import os
import keras

from flask import Flask,request,jsonify
from flask_restful import Resource, Api, reqparse
import numpy as np
import cv2
from PIL import Image
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)


num_classes = 10

model = keras.models.load_model(
    MODEL_PATH
)
app = Flask(__name__)
api = Api(app)

@app.route('/home', methods=['POST'])
def home():
    data = request.files['file']
    pil_image = Image.open(data.stream)
    # img = cv2.imread(data,cv2.IMREAD_GRAYSCALE)
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2GRAY)
    pred = model.predict(img.reshape(1,28,28,1))
    result = pred.argmax()
    print("Predicted output:", pred.argmax())   
    return jsonify({"message" :str(result)}),200



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
