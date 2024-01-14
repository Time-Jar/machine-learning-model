from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf

# supervised ML algorithm
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
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Output Layer for Percentage (regression)
    output = Dense(1, activation='sigmoid')(x) # Binary classification
        
    # Model
    model: Model = Model(inputs=list(inputs.values()), outputs=output)

    # Compile the model (the loss function)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Binary classification
    
    return model

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Multiply
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def attention_mechanism(embeddings):
    # Concatenate embeddings
    concat_embeddings = Concatenate()(embeddings)
    
    # Calculate attention scores
    attention_scores = Dense(concat_embeddings.shape[1], activation='softmax')(concat_embeddings)
    
    # Apply attention scores
    weighted_embeddings = Multiply()([concat_embeddings, attention_scores])

    # Combine the embeddings using a sum operation
    combined_embeddings = tf.reduce_sum(weighted_embeddings, axis=1)

    return combined_embeddings


def build_and_compile_model_with_attention_machanism(num_buckets, embedding_dim, feature_columns):
    # Input Layers
    inputs = {col: Input(shape=(1,), name=col) for col in feature_columns}

    # Embedding for hash-encoded columns
    embeddings = []
    for col in ['user_id', 'app_name']:
        emb = Embedding(input_dim=num_buckets, output_dim=embedding_dim, input_length=1)(inputs[col])
        emb = Flatten()(emb)
        embeddings.append(emb)

    # Apply Attention Mechanism
    attention_output = attention_mechanism(embeddings)

    # Ensure attention_output is 2D (if it's not already)
    attention_output_2d = tf.expand_dims(attention_output, -1)  # Expand dimensions if necessary

    # Flatten or reshape other_cols for compatibility
    other_cols = [inputs[col] for col in feature_columns if col not in ['user_id', 'app_name']]
    reshaped_other_cols = [Flatten()(col) for col in other_cols]  # Flatten each column

    # Concatenated features
    concatenated_features = Concatenate(axis=-1)([attention_output_2d] + reshaped_other_cols)

    # Dense Layers (kernel_regularizer to prevent overfitting)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(concatenated_features)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # Output Layer for Binary Classification
    output = Dense(1, activation='sigmoid')(x)

    # Model
    model = Model(inputs=list(inputs.values()), outputs=output)
    
    # Learning rate
    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000000,
        decay_rate=0.90,
        staircase=True
    )

    # Compile the model for Binary Classification
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

    return model

## Other

import os

def get_file_size_in_mb(file_path):
    # Get file size in bytes
    file_size_bytes = os.path.getsize(file_path)

    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)

    return file_size_mb