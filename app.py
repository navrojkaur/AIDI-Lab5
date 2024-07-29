# app.py
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    length = request.form['length']
    # Extract other features from the form
    # Make prediction
    prediction = model.predict([[length]])  # Adjust based on the model and features
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
