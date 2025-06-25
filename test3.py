import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error


# Step 1: Load and preprocess data
def load_data(train_path, test_path, rul_path):
    """
    Load and preprocess the CMAPSS dataset.
    Args:
        train_path: Path to the training dataset.
        test_path: Path to the testing dataset.
        rul_path: Path to the Remaining Useful Life (RUL) data.
    Returns:
        train_df: Training dataset with RUL calculated.
        test_df: Testing dataset with 'RUL' column added (filled with NaN).
        rul_df: Remaining Useful Life (RUL) data for the test dataset.
    """
    train_df = pd.read_csv(train_path, sep=" ", header=None).dropna(axis=1, how='all')
    test_df = pd.read_csv(test_path, sep=" ", header=None).dropna(axis=1, how='all')
    rul_df = pd.read_csv(rul_path, sep=" ", header=None).dropna(axis=1, how='all')

    columns = ['unit_number', 'time_in_cycles'] + [f'operational_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    train_df.columns = columns
    test_df.columns = columns
    rul_df.columns = ['RUL']

    # Add RUL to train dataset
    max_cycle = train_df.groupby('unit_number')['time_in_cycles'].max()
    train_df['RUL'] = train_df.apply(lambda x: max_cycle[x['unit_number']] - x['time_in_cycles'], axis=1)

    # Add a placeholder RUL column to the test dataset (with NaN or zeros)
    test_df['RUL'] = np.nan  # or use 0 if preferred

    return train_df, test_df, rul_df


# Step 2: Normalize sensor data
def normalize_data(train_df, test_df, sensor_cols):
    """
    Normalize the sensor data using MinMaxScaler.
    Args:
        train_df: Training dataset.
        test_df: Testing dataset.
        sensor_cols: List of sensor column names to normalize.
    Returns:
        train_df: Normalized training dataset.
        test_df: Normalized testing dataset.
        scaler: Fitted MinMaxScaler instance.
    """
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    return train_df, test_df, scaler


# Step 3: Prepare sequences for LSTM model
def prepare_sequences(data, sequence_length, sensor_cols, target_col):
    """
    Prepare sequences for LSTM training.
    Args:
        data: Dataset to prepare sequences from.
        sequence_length: Length of the sequences.
        sensor_cols: List of sensor column names.
        target_col: Target column name.
    Returns:
        sequences: Array of input sequences.
        targets: Array of target values.
    """
    sequences = []
    targets = []
    for unit in data['unit_number'].unique():
        unit_data = data[data['unit_number'] == unit]
        for i in range(len(unit_data) - sequence_length):
            seq = unit_data.iloc[i:i + sequence_length][sensor_cols].values
            target = unit_data.iloc[i + sequence_length - 1][target_col]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)


# Step 4: Train Autoencoder for dimensionality reduction
def train_autoencoder(train_data, sensor_cols, encoding_dim=10):
    """
    Train an autoencoder for dimensionality reduction.
    Args:
        train_data: Training dataset.
        sensor_cols: List of sensor column names.
        encoding_dim: Dimensionality of the encoded layer.
    Returns:
        autoencoder: Trained autoencoder model. 
        encoder: Encoder model to extract reduced features.
    """
    input_dim = len(sensor_cols)
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    autoencoder.fit(
        train_data[sensor_cols], train_data[sensor_cols],
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_split=0.2
    )

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[2].output)
    return autoencoder, encoder


# Step 5: Train LSTM model for RUL prediction
def train_lstm(train_sequences, train_targets, val_split=0.2):
    """
    Train an LSTM model for RUL prediction.
    Args:
        train_sequences: Input sequences for training.
        train_targets: Target values for training.
        val_split: Fraction of data to use for validation.
    Returns:
        model: Trained LSTM model.
    """
    model = Sequential([ 
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(train_sequences.shape[1], train_sequences.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(1, activation='relu')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.fit(
        train_sequences, train_targets,
        epochs=20,
        batch_size=64,
        validation_split=val_split
    )
    return model


# Step 6: Save models
def save_models(autoencoder, lstm_model, autoencoder_path='autoencoder_model.h5', lstm_model_path='lstm_model.h5'):
    """
    Save the trained autoencoder and LSTM model to specified file paths.
    Args:
        autoencoder: Trained autoencoder model.
        lstm_model: Trained LSTM model.
        autoencoder_path: Path to save the autoencoder model.
        lstm_model_path: Path to save the LSTM model.
    """
    autoencoder.save(autoencoder_path)
    lstm_model.save(lstm_model_path)
    print(f"Models saved: {autoencoder_path}, {lstm_model_path}")


# Main pipeline function
def main(train_path, test_path, rul_path, sequence_length=50):
    # Load and preprocess the data
    train_df, test_df, rul_df = load_data(train_path, test_path, rul_path)

    # Define the columns for sensors
    sensor_cols = [col for col in train_df.columns if 'sensor' in col]

    # Normalize the data
    train_df, test_df, scaler = normalize_data(train_df, test_df, sensor_cols)

    # Train an autoencoder and extract encoded features
    autoencoder, encoder = train_autoencoder(train_df, sensor_cols, encoding_dim=10)
    encoded_features = encoder.predict(train_df[sensor_cols])

    # Add encoded features to the DataFrame
    encoded_feature_cols = [f'encoded_{i}' for i in range(encoded_features.shape[1])]
    for i, col in enumerate(encoded_feature_cols):
        train_df[col] = encoded_features[:, i]

    # Prepare sequences for LSTM
    train_sequences, train_targets = prepare_sequences(train_df, sequence_length, sensor_cols, 'RUL')
    test_sequences, test_targets = prepare_sequences(test_df, sequence_length, sensor_cols, 'RUL')

    # Train LSTM model
    lstm_model = train_lstm(train_sequences, train_targets)

    # Save the trained models
    save_models(autoencoder, lstm_model)

    return autoencoder, lstm_model


# Run the main function and specify paths
train_path = r"dataset\train_FD001.txt"
test_path = r"dataset\test_FD001.txt"
rul_path = r"dataset\RUL_FD001.txt"

autoencoder, lstm_model = main(train_path, test_path, rul_path)
