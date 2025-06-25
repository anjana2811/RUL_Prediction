import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# ✅ Step 1: Load and preprocess data
def load_data(train_path, test_path, rul_path):
    train_df = pd.read_csv(train_path, sep=" ", header=None).dropna(axis=1, how="all")
    test_df = pd.read_csv(test_path, sep=" ", header=None).dropna(axis=1, how="all")
    rul_df = pd.read_csv(rul_path, sep=" ", header=None).dropna(axis=1, how="all")

    columns = ["unit_number", "time_in_cycles"] + [f"operational_setting_{i}" for i in range(1, 4)] + \
              [f"sensor_{i}" for i in range(1, 22)]
    train_df.columns = columns
    test_df.columns = columns
    rul_df.columns = ["RUL"]

    # Add RUL to train dataset
    max_cycle = train_df.groupby("unit_number")["time_in_cycles"].max()
    train_df["RUL"] = train_df.apply(lambda x: max_cycle[x["unit_number"]] - x["time_in_cycles"], axis=1)

    # Merge RUL with test_df
    test_df = test_df.merge(rul_df, left_on="unit_number", right_index=True, how="left")

    return train_df, test_df, rul_df

# ✅ Step 2: Normalize sensor data
def normalize_data(train_df, test_df, sensor_cols, scaler_path="scaler.npy"):
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    # ✅ Save the scaler correctly
    np.save(scaler_path, {"min": scaler.min_, "scale": scaler.scale_}, allow_pickle=True)

    return train_df, test_df, scaler

# ✅ Step 3: Train Autoencoder and Extract Features
def train_autoencoder(train_data, sensor_cols, encoding_dim=10):
    input_layer = Input(shape=(len(sensor_cols),))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu', name="encoded_layer")(encoded)  
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(len(sensor_cols), activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    autoencoder.fit(
        train_data[sensor_cols], train_data[sensor_cols],
        epochs=50, batch_size=256, shuffle=True, validation_split=0.2
    )

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("encoded_layer").output)
    return autoencoder, encoder

# ✅ Step 4: Prepare sequences for LSTM model
def prepare_sequences(data, sequence_length, feature_cols, target_col):
    sequences, targets = [], []
    for unit in data['unit_number'].unique():
        unit_data = data[data['unit_number'] == unit]
        for i in range(len(unit_data) - sequence_length):
            seq = unit_data.iloc[i:i + sequence_length][feature_cols].values
            target = unit_data.iloc[i + sequence_length - 1][target_col]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)

# ✅ Step 5: Train LSTM model with training history plot
def train_lstm(train_sequences, train_targets):
    model = Sequential([
        Bidirectional(LSTM(64, activation='tanh', return_sequences=True, input_shape=(train_sequences.shape[1], train_sequences.shape[2]))),
        Dropout(0.2),
        Bidirectional(LSTM(32, activation='tanh')),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # ✅ Store training history
    history = model.fit(train_sequences, train_targets, epochs=30, batch_size=64, validation_split=0.2)

    # ✅ Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.show()

    return model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ✅ Step 6: Evaluate Model Performance
def evaluate_model(lstm_model, test_sequences, test_targets):
    predictions = lstm_model.predict(test_sequences).flatten()

    # ✅ Convert to NumPy arrays and remove NaN values
    test_targets = np.array(test_targets)
    predictions = np.array(predictions)

    valid_indices = ~np.isnan(test_targets) & ~np.isnan(predictions)
    test_targets = test_targets[valid_indices]
    predictions = predictions[valid_indices]

    # ✅ Compute performance metrics
    mae = mean_absolute_error(test_targets, predictions)
    mse = mean_squared_error(test_targets, predictions)
    rmse = np.sqrt(mse)  # Manually compute RMSE
    r2 = r2_score(test_targets, predictions)

    print(f"✅ Model Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} cycles")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} cycles")
    print(f"R² Score: {r2:.2f}")

    # ✅ Convert RUL into categories for confusion matrix
    def categorize_rul(rul):
        if rul < 50:
            return 0  # High Risk of Failure
        elif 50 <= rul <= 150:
            return 1  # Moderate Risk
        else:
            return 2  # Healthy

    actual_classes = np.array([categorize_rul(r) for r in test_targets])
    predicted_classes = np.array([categorize_rul(r) for r in predictions])

    # ✅ Generate Confusion Matrix
    conf_matrix = confusion_matrix(actual_classes, predicted_classes)

    # ✅ Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["High Risk", "Moderate", "Healthy"], yticklabels=["High Risk", "Moderate", "Healthy"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for RUL Classification")
    plt.show()

    # ✅ Print Classification Report
    print("\nClassification Report:")
    print(classification_report(actual_classes, predicted_classes, target_names=["High Risk", "Moderate", "Healthy"]))

    # ✅ Plot Actual vs. Predicted RUL
    plt.figure(figsize=(10, 5))
    plt.scatter(test_targets, predictions, alpha=0.6, label="Predictions")
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], color='red', linestyle='--', label="Perfect Fit")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Actual vs. Predicted RUL")
    plt.legend()
    plt.show()

    # ✅ Plot Histogram of Errors
    errors = test_targets - predictions
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Prediction Error (RUL)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Errors")
    plt.show()

# ✅ Step 7: Save models
def save_models(autoencoder, lstm_model, scaler):
    autoencoder.save("autoencoder_model.h5", save_format="h5", include_optimizer=False)
    lstm_model.save("lstm_model.h5", save_format="h5", include_optimizer=False)

    np.save("scaler.npy", {"min": scaler.min_, "scale": scaler.scale_}, allow_pickle=True)

# ✅ Step 8: Main Function
def main(train_path, test_path, rul_path, sequence_length=50):
    train_df, test_df, _ = load_data(train_path, test_path, rul_path)

    sensor_cols = [col for col in train_df.columns if "sensor" in col]
    train_df, test_df, scaler = normalize_data(train_df, test_df, sensor_cols)

    autoencoder, encoder = train_autoencoder(train_df, sensor_cols)
    encoded_train = encoder.predict(train_df[sensor_cols])
    encoded_test = encoder.predict(test_df[sensor_cols])

    encoded_feature_cols = [f"encoded_{i}" for i in range(encoded_train.shape[1])]
    for i, col in enumerate(encoded_feature_cols):
        train_df[col] = encoded_train[:, i]
        test_df[col] = encoded_test[:, i]

    train_sequences, train_targets = prepare_sequences(train_df, sequence_length, encoded_feature_cols, "RUL")
    test_sequences, test_targets = prepare_sequences(test_df, sequence_length, encoded_feature_cols, "RUL")

    lstm_model = train_lstm(train_sequences, train_targets)
    
    # ✅ Evaluate model
    evaluate_model(lstm_model, test_sequences, test_targets)

    save_models(autoencoder, lstm_model, scaler)

# ✅ Define Paths
train_path = r"C:\FINAL_RUL\dataset\train_FD001.txt"
test_path = r"C:\FINAL_RUL\dataset\test_FD001.txt"
rul_path = r"C:\FINAL_RUL\dataset\RUL_FD001.txt"

# ✅ Run Training
main(train_path, test_path, rul_path)
 