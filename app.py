from flask import Flask, request, render_template
import os
from predict.predicting import predict
app = Flask(__name__)
UPLOAD_FOLDER = "E:/BeCodeProjects/Mole_Detection_Project/static/"


@app.route('/', methods=['GET', 'POST'])
def upload():
    return render_template('mole_predict.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            #print(image_location)
            image_file.save(image_location)
            status = predict(image_location)
            #print(status)
            return render_template('mole_predict.html', prediction=status)

    return render_template('mole_predict.html')


if __name__ == '__main__':
    app.run(port=5000)
