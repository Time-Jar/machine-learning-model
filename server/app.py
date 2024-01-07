from flask import Flask, request, jsonify
import tensorflow as tf
import os

app = Flask(__name__)

# Load the pre-trained model (Ensure this path is correct)
model = tf.keras.models.load_model(os.environ.get("MODEL_PATH"))

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.json

    # Process the data as per the model's requirement
    # ...
    # For example:
    # user_app_usage = preprocess(data['user_app_usage'])
    # users_data = preprocess(data['users_data'])

    # Make a prediction
    prediction = model.predict([user_app_usage, users_data])

    # Convert the prediction to percentage
    prediction_percent = prediction * 100

    # Return the result
    return jsonify({'prediction': prediction_percent.tolist()})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
