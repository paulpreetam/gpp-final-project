# Importing essential libraries
from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the Random Forest CLassifier model
filename = 'heart_attack_prediction_rfc_model.pkl'
scaler_name = 'scaler.pkl'
classifier = joblib.load(filename)
scaler = joblib.load(scaler_name)

app = Flask(__name__) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predictions', methods=['POST'])
def predictions():
     if request.method == 'POST':
        age = int(request.form['age'])
        # sex = int(request.form['sex'])
        cp = int(request.form['chest_pain'])
        tr = int(request.form['blood_pressure'])
        chol = int(request.form['cholesterol'])
        # fbs = int(request.form['blood_sugar'])
        # ecg = int(request.form['cardiographic_results'])
        chh = int(request.form['heart_rate'])
        # exng = int(request.form['angina'])
        peak = float(request.form['previous_peak'])
        # slp = int(request.form['slope'])
        caa = int(request.form['vessels'])
        thall = int(request.form['thal'])

        data = np.array([[age, cp, tr, chol, chh, peak,\
            caa, thall]])
        
        scaled_data = scaler.transform(data)
        prediction = classifier.predict(scaled_data)
        
        return render_template('result.html', prediction=prediction)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


if __name__ == '__main__':

    # Run this when running on LOCAL server...
    # app.run(debug=True)

    # ...OR run this when PRODUCTION server.
    app.run(debug=False)