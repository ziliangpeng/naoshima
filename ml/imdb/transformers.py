import tensorflow as tf
import numpy as np

"""
Written by GPT. Not working yet.
"""

# Generate synthetic data (Replace this with real data)
X = ["This is good", "This is bad", "I love it", "I hate it"]
y = [1, 0, 1, 0]  # 1 is positive, 0 is negative

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_sequences, padding="post")

# Convert to TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_padded, y)).batch(2)

# Model Hyperparameters
num_layers = 2
d_model = 64
num_heads = 4
dff = 128
input_vocab_size = 5000
max_position_encoding = 100


# Positional encoding for transformer
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# Transformer Encoder Layer
def encoder_layer(d_model, num_heads, dff, rate=0.1):
    inputs = tf.keras.Input(shape=(None, d_model))
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(
        inputs, inputs
    )
    attn = tf.keras.layers.Dropout(rate)(attn)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn)

    ffn_output = tf.keras.layers.Dense(dff, activation="relu")(out1)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)

    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return tf.keras.Model(inputs=inputs, outputs=out2)


# Encoder consisting of multiple Encoder Layers
def encoder(
    vocab_size, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1
):
    inputs = tf.keras.Input(shape=(None,))
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += positional_encoding(maximum_position_encoding, d_model)

    x = tf.keras.layers.Dropout(rate)(x)

    for _ in range(num_layers):
        x = encoder_layer(d_model, num_heads, dff, rate)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# Classifier model using Transformer Encoder
def transformer_classifier(
    vocab_size,
    num_layers,
    d_model,
    num_heads,
    dff,
    maximum_position_encoding,
    rate=0.1,
    num_classes=2,
):
    inputs = tf.keras.Input(shape=(None,))
    x = encoder(
        vocab_size, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate
    )(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Create classifier model
model = transformer_classifier(
    input_vocab_size, num_layers, d_model, num_heads, dff, max_position_encoding
)

# Compile and Train
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(dataset, epochs=50)
