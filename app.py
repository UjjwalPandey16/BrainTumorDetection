import os

import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('models/brainTumor-4category-b64e50-categorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_classname(classname):
    if classname[0][0] == 1:
        return "Glioma Tumor Present."
    elif classname[0][1] == 1:
        return "Meningioma Tumor Present"
    elif classname[0][2] == 1:
        return "No Tumor present."
    elif classname[0][3] == 1:
        return "Pituitary Tumor Present"


def getresult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getresult(file_path)
        result = get_classname(value)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
