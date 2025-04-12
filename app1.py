from flask import Flask, render_template, request
import keras
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing import image
import os
import cv2
import efficientnet.tfkeras as efn

app = Flask(__name__, template_folder='templates', static_folder='css')
model = load_model('Resnet.h5')
target_img = os.path.join(os.getcwd(), 'css/images')


@app.route('/')
def index():
    return render_template('index.html')


ALLOWED_EXT = set(['jpg', 'png', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT


def resize_image(image_path, size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (size, size))
    return resized_img


def read_img(filename):
    img = load_img(filename, target_size=(128, 128))
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    # x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def check_image_matching(template, query, threshold=0.5):
    # Perform template matching
    result = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the match is above the threshold
    if max_val >= threshold:
        return True
    else:
        return False


@app.route('/protect', methods=["GET", "POST"])
def protect():
    file = request.files['img']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join('css/images', filename)
        file.save(file_path)
        img = read_img(file_path)

        # Replace with your template image path
        template_path = 'css/images/a1.png'
        template = resize_image(template_path, 128)
        query = resize_image(file_path, 128)
        if check_image_matching(template, query):
            pred = model.predict(img)
            pred = np.argmax(pred, axis=1)
            if pred[0] == 2:
                data = "NO DIABETIC RETINOPATHY"
            elif pred[0] == 0:
                data = "MILD DIABETIC RETINOPATHY"
            elif pred[0] == 1:
                data = "MODERATE DIABETIC RETINOPATHY"
            elif pred[0] == 4:
                data = "SEVERE DIABETIC RETINOPATHY"
            elif pred[0] == 3:
                data = "PROLIFERATIVE DIABETIC RETINOPATHY"
            return render_template('protect.html', data=data)
        else:
            return render_template('protect.html', data="Please Provide Retina images")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
