import pandas as pd
from datetime import datetime
import hashlib

# Convert to pandas DataFrames
def toPandasDataframes(_acceptance_data, _actions_data, _app_names_data, _location_data, _sex, _weekdays, user_app_usage_data, users_data):
    df__acceptance = pd.DataFrame(_acceptance_data)
    df__actions = pd.DataFrame(_actions_data)
    df__app_names = pd.DataFrame(_app_names_data)
    df__location = pd.DataFrame(_location_data)
    df__sex = pd.DataFrame(_sex)
    df__weekdays = pd.DataFrame(_weekdays)

    df_user_app_usage = pd.DataFrame(user_app_usage_data)
    df_users = pd.DataFrame(users_data)
    
    return df__acceptance, df__actions, df__app_names, df__location, df__sex, df__weekdays, df_user_app_usage, df_users

# Calculate/simplify data functions

def convert_boolean_to_numeric(df_original, column_name):
    """
    Converts a boolean column in a DataFrame to 0 or 1.
    """
    df = df_original.copy()

    # Convert boolean to int (True to 1, False to 0)
    df[column_name] = df[column_name].astype(int)

    return df

def convert_string_to_date(df_original, dob_column):
    """
    Converts a date of birth column from string to datetime and calculates the age.
    """
    df = df_original.copy()
    df[dob_column] = pd.to_datetime(df[dob_column])
    df['age'] = df[dob_column].apply(
        lambda dob: datetime.now().year - dob.year - ((datetime.now().month, datetime.now().day) < (dob.month, dob.day))
    )
    return df

## Final functions

def normalize_numerical_data(df_original, column, fixed_max):
    """
    Normalizes a specified column of the DataFrame.
    """
    df = df_original.copy()

    # Normalize the column
    df[column] = df[column] / fixed_max
    
    # Ensure that the values do not exceed 1, more than 1 are clipped to 1
    df[column] = df[column].clip(lower=0, upper=1)
    
    return df

def one_hot_encoding(df_original: pd.DataFrame, map_df: pd.DataFrame, column_to_encode: str, map_column: str, map_values: str):
    """
    Maps a column to new values and applies one-hot encoding, ensuring all categories are represented.
    """
    df = df_original.copy()
    
    # Map the column to new valuesthan
    mapping = dict(zip(map_df[map_column], map_df[map_values]))
    df[column_to_encode] = df[column_to_encode].map(mapping)

    # One-hot encoding
    df = pd.get_dummies(df, columns=[column_to_encode], prefix=column_to_encode)

    # Add missing columns (if any) and fill with 0
    required_columns = map_df[map_values].unique()
    for col in required_columns:
        full_col_name = f'{column_to_encode}_{col}'
        if full_col_name not in df.columns:
            df[full_col_name] = 0
        df[full_col_name] = df[full_col_name].astype(int)

    return df

def normalize_time(df_original, time_column):
    """
    Normalizes time values in a DataFrame column.
    """
    df = df_original.copy()
    
    # Convert the time column to pandas datetime
    df[time_column] = pd.to_datetime(df[time_column])

    # Normalize time: hour + minute/60 + second/3600, then divide by 24
    df[time_column] = df[time_column].apply(lambda x: (x.hour + x.minute / 60 + x.second / 3600) / 24)

    return df

def hash_encode(df_original: pd.DataFrame, column: str, num_buckets: int):
    """
    Hash-encodes a column in a DataFrame.
    """
    df = df_original.copy()
    
    def hash_column(data):
        return int(hashlib.sha256(str(data).encode()).hexdigest(), 16) % num_buckets

    df[column] = df[column].apply(hash_column)

    return df

# Merging data

import re

# Remember, this is the final processed data already
def set_column_data_types(df_original):
    """
    Set data types for specific columns of a pandas DataFrame based on predefined rules.
    """
    
    df = df_original.copy()

    # Combined column patterns and their corresponding data types
    column_type_patterns = {
        'float32': ['age', 'time_of_day', 'app_usage_time'],
        'bool': ['sex_', 'should_be_blocked', 'weekday_', 'acceptance_', 'action_', 'location_'],
        'uint16': ['app_name', 'user_id']
    }

    for dtype, patterns in column_type_patterns.items():
        for pattern in patterns:
            matching_columns = [col for col in df.columns if re.match(pattern, col) or col == pattern]
            for col in matching_columns:
                df[col] = df[col].astype(dtype)

    return df