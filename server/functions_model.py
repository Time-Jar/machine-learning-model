from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model

def build_and_compile_model(num_buckets, embedding_dim, feature_columns):
    """
    Builds and compiles a TensorFlow model for the given feature columns.
    """
    # Input Layers
    inputs = {col: Input(shape=(1,), name=col) for col in feature_columns}

    # Embedding for hash-encoded columns
    embeddings = []
    for col in ['user_id', 'app_name']:
        emb = Embedding(input_dim=num_buckets, output_dim=embedding_dim, input_length=1)(inputs[col])
        emb = Flatten()(emb)
        embeddings.append(emb)

    # Directly use other columns
    other_cols = [inputs[col] for col in feature_columns if col not in ['user_id', 'app_name']]
    concatenated_features = Concatenate()(embeddings + other_cols)

    # For simplicity, using Dense layers instead of Transformer
    x = Dense(128, activation='relu')(concatenated_features)
    x = Dense(64, activation='relu')(x)

    # Output Layer for Percentage (regression)
    output = Dense(1, activation='linear')(x)  # 'linear' can be omitted as it is the default

    # Model
    model: Model = Model(inputs=list(inputs.values()), outputs=output)

    # Compile the model for regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # mae is mean absolute error

    return model