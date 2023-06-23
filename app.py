from flask import Flask
# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

# Load the Random Forest CLassifier model
filename = 'model/xgb_clf.pkl'
classifier = joblib.load(open(filename, 'rb'))

filename = 'model/robust_scaler.pkl'
scaler = joblib.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])

        transformed_data = scaler.transform(data)

        my_prediction = classifier.predict(transformed_data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)