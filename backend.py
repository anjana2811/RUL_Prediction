from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import MinMaxScaler
import os
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Use the original paths from your code
AUTOENCODER_PATH = r"C:\FINAL_RUL\autoencoder_model.h5"
LSTM_MODEL_PATH = r"C:\FINAL_RUL\lstm_model.h5"
SCALER_PATH = r"C:\FINAL_RUL\scaler.npy"

# Load models and scaler within a function to handle errors properly
def load_models():
    try:
        autoencoder = load_model(AUTOENCODER_PATH, compile=False)
        lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
        scaler_params = np.load(SCALER_PATH, allow_pickle=True).item()
        scaler = MinMaxScaler()
        scaler.min_, scaler.scale_ = scaler_params["min"], scaler_params["scale"]
        print("Models and scaler loaded successfully!")
        return autoencoder, lstm_model, scaler
    except Exception as e:
        print(f"Error loading models or scaler: {e}")
        return None, None, None

autoencoder, lstm_model, scaler = load_models()

def parse_file(file):
    """Parse uploaded file regardless of whether it's CSV or TXT format"""
    # Read file content
    content = file.read().decode('utf-8')
    file.seek(0)  # Reset file pointer
    
    try:
        # Try reading as a space-separated file (works for both txt and csv with space delimiter)
        df = pd.read_csv(io.StringIO(content), sep=" ", header=None).dropna(axis=1, how="all")
        
        # If file is empty or couldn't be parsed properly
        if df.empty or len(df.columns) < 24:  # We expect at least 24 columns
            raise ValueError("File format not recognized or insufficient data columns")
            
        return df
    except Exception as e:
        # Try alternative parsing methods if the first method fails
        try:
            # Try comma-separated format
            df = pd.read_csv(io.StringIO(content), header=None).dropna(axis=1, how="all")
            if df.empty or len(df.columns) < 24:
                raise ValueError("File format not recognized as CSV")
            return df
        except Exception:
            # Try tab-separated format
            try:
                df = pd.read_csv(io.StringIO(content), sep="\t", header=None).dropna(axis=1, how="all")
                if df.empty or len(df.columns) < 24:
                    raise ValueError("File format not recognized as TSV")
                return df
            except Exception:
                raise ValueError(f"Could not parse file: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict_rul():
    if None in (autoencoder, lstm_model, scaler):
        return jsonify({"error": "Models not loaded correctly. Check server logs."}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Check file extension
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.txt')):
        return jsonify({"error": "Only CSV and TXT files are supported"}), 400
    
    try:
        df = parse_file(file)
    except Exception as e:
        return jsonify({"error": f"Error parsing file: {str(e)}"}), 400
    
    column_names = ["engine_id", "time_in_cycles"] + [f"operational_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    
    # Ensure the dataframe has enough columns
    if len(df.columns) < len(column_names):
        return jsonify({"error": f"File does not contain enough columns. Expected at least {len(column_names)}, got {len(df.columns)}"}), 400
    
    # Use only the columns we need
    df = df.iloc[:, :len(column_names)]
    df.columns = column_names
    
    try:
        engine_id = int(df.iloc[0]["engine_id"])
    except (ValueError, KeyError):
        return jsonify({"error": "Could not extract engine ID from file"}), 400
    
    df_last_50 = df.iloc[-50:] if len(df) >= 50 else df
    time_steps = df_last_50["time_in_cycles"].tolist()
    df_sensors = df_last_50[[col for col in df.columns if "sensor" in col]]
    
    try:
        normalized_sensor_data = scaler.transform(df_sensors)
    except Exception as e:
        return jsonify({"error": f"Error normalizing sensor data: {str(e)}"}), 400
    
    # Get the encoder layer safely
    try:
        encoder_layer = autoencoder.get_layer("encoded_layer")
    except ValueError:
        # If the exact layer name is wrong, get the bottleneck layer (assuming it's the middle layer)
        encoder_layer = autoencoder.layers[len(autoencoder.layers) // 2]
    
    encoder_model = Model(inputs=autoencoder.input, outputs=encoder_layer.output)
    
    try:
        encoded_features = encoder_model.predict(normalized_sensor_data)
    except Exception as e:
        return jsonify({"error": f"Error encoding features: {str(e)}"}), 400
    
    # Ensure the encoded features match the expected input shape of LSTM model
    expected_feature_size = lstm_model.input_shape[-1]
    
    try:
        lstm_input = encoded_features.reshape(1, len(df_last_50), expected_feature_size)
        predicted_rul = float(lstm_model.predict(lstm_input)[0][0])
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 400
    
    # Create sensor contribution data for frontend visualization
    sensor_data = []
    for i, col in enumerate(df_sensors.columns):
        # Calculate average value for each sensor as a simple contribution metric
        sensor_data.append({
            "sensor": col,
            "value": float(df_sensors[col].mean())
        })
    
    # Create time series data for RUL trend visualization
    time_series_data = []
    for i, step in enumerate(time_steps):
        time_series_data.append({
            "cycle": int(step),
            "rul": predicted_rul - (len(time_steps) - i - 1)  # Simple linear decrease
        })
    
    return jsonify({
        "engine_id": engine_id,
        "predicted_rul": round(predicted_rul, 2),  # Round to 2 decimal places
        "time_steps": time_steps,
        "graphData": sensor_data,  # Add data for chart visualization
        "rulTrend": time_series_data,  # Add RUL trend data
        "matched_cycle": int(df_last_50["time_in_cycles"].iloc[-1])  # Add matched cycle
    })

if __name__ == '__main__':
    app.run(debug=True)