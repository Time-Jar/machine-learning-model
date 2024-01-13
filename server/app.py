from flask import Flask, request, jsonify
from numpy import uint8
import tensorflow as tf
import os
from . import functions_supabase, functions_aggregated, functions_basic

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model(os.environ.get("MODEL_PATH") or "./model/model.keras") # type: ignore
import re

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from request
        data = request.json
        
        if (data == None):
            raise ValueError("request json not present")
        
        userId = data['userId']
        appNameId = data['appNameId']
        weekday = data['weekday']
        timeOfDay = data['timeOfDay']
        locationId = data['locationId']
        
        supabase = functions_supabase.auth()

        _acceptance_data, _actions_data, _app_names_data, _location_data, _sex, _weekdays, user_app_usage_data, users_data = functions_supabase.fetchTables(supabase, False)
        user_app_usage_data = [
            {
                'id': 0, # removed later
                'created_at': "", # removed later
                'app_name': appNameId,
                'user_id': userId,
                'acceptance': -1, # removed later
                'should_be_blocked': False, # removed later
                'action': -1, # removed later
                'location': locationId,
                'weekday': weekday,
                'time_of_day': timeOfDay,
                'app_usage_time': 0 # removed later
            },
        ]
        
        df__acceptance, df__actions, df__app_names, df__location, df__sex, df__weekdays, df_user_app_usage, df_users = functions_basic.toPandasDataframes(_acceptance_data, _actions_data, _app_names_data, _location_data, _sex, _weekdays, user_app_usage_data, users_data)
        df_user_app_usage_normalized, df_users_normalized = functions_aggregated.normalizeAndNumericalize(df__acceptance, df__actions, df__app_names, df__location, df__sex, df__weekdays, df_user_app_usage, df_users)
        merged_df = functions_aggregated.mergeUsersAndAppUsage(df_user_app_usage_normalized, df_users_normalized)

        # To clear
        patterns_to_clear = ['acceptance_.*', 'action_.*'] # Patterns for columns to be dropped
        columns_matching_patterns = [col for col in merged_df.columns if any(re.match(pattern, col) for pattern in patterns_to_clear)] # Find columns that match the patterns

        # Drop and clear the columns
        merged_df = merged_df.drop(columns=['should_be_blocked'])
        merged_df[columns_matching_patterns] = False
        merged_df['app_usage_time'] = -1
        
        merged_df['app_usage_time'] = merged_df['app_usage_time'].astype('float32')

        # Loop through each column and check its data type
        #for column in merged_df.columns:
        #    if merged_df[column].dtype == 'bool' or column in all_columns_to_clear:
        #        # Convert boolean columns to uint8 type
        #        merged_df[column] = merged_df[column].astype('bool')

        print(merged_df)
        print(merged_df.dtypes)

        input_tensors = [merged_df[column].values for column in merged_df.columns]

        # Make a prediction
        prediction = model.predict(input_tensors)
        
        print("prediction", prediction)

        # Return the result
        return jsonify({'prediction': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': e})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
