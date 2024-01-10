from flask import Flask, request, jsonify
import tensorflow as tf
import os
from server import functions_supabase, functions_aggregated, functions_basic

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model(os.environ.get("MODEL_PATH") or "./model/model.keras") # type: ignore

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from request
        data = request.json
        
        if (data == None):
            raise ValueError("request json not present")
        
        userId = data['userId']
        
        supabase = functions_supabase.auth()

        _acceptance_data, _actions_data, _app_names_data, _location_data, _sex, _weekdays, user_app_usage_data, users_data = functions_supabase.fetchTables(supabase, userId)
        df__acceptance, df__actions, df__app_names, df__location, df__sex, df__weekdays, df_user_app_usage, df_users = functions_basic.toPandasDataframes(_acceptance_data, _actions_data, _app_names_data, _location_data, _sex, _weekdays, user_app_usage_data, users_data)
        df_user_app_usage_normalized, df_users_normalized = functions_aggregated.normalizeAndNumericalize(df__acceptance, df__actions, df__app_names, df__location, df__sex, df__weekdays, df_user_app_usage, df_users)
        merged_df = functions_aggregated.mergeUsersAndAppUsage(df_user_app_usage_normalized, df_users_normalized)

        print(merged_df)

        # Make a prediction
        prediction = model.predict(merged_df)
        
        print(prediction)

        # Convert the prediction to percentage
        prediction_percent = prediction * 100

        # Return the result
        return jsonify({'prediction': prediction_percent})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
