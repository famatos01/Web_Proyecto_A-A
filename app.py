# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar modelo y codificadores
model = joblib.load('car_model.pkl')
encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    features = {
        'buying': request.form['buying'],
        'maint': request.form['maint'],
        'doors': request.form['doors'],
        'persons': request.form['persons'],
        'lug_boot': request.form['lug_boot'],
        'safety': request.form['safety']
    }
    
    # Codificar datos
    encoded_data = {}
    for col, value in features.items():
        encoded_data[col] = encoders[col].transform([value])[0]
    
    # Predecir
    prediction = model.predict([list(encoded_data.values())])[0]
    class_name = encoders['class'].inverse_transform([prediction])[0]
    
    return render_template('result.html', 
                         prediction=class_name,
                         features=features)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
    