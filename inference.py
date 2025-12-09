"""
Inference Script for Engine RUL Prediction with Explainable AI
Outputs: engine_failure_reports.csv with detailed analysis
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ===================== Configuration =====================
TRAIN_PATH = 'train_FD001.txt'
TEST_PATH = 'test_FD001.txt'
TRUTH_PATH = 'RUL_FD001.txt'
MODEL_PATH = 'hp_search_results/best_lr0.0005_lam5.0_ch64.pth'
OUTPUT_CSV = 'report.csv'

SEQ_LEN = 30
MAX_RUL = 125
CNN_CHANNELS = 64  # From best model configuration
THRESHOLD_RUL_FLAG = 30  # Threshold for imminent failure warning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== Sensor to Subsystem Mapping =====================
SENSOR_TO_SUBSYSTEM = {
    'sensor_1': ['Fan inlet temperature', 'Overall Inlet'],
    'sensor_2': ['Fan inlet pressure (psia)', 'Overall Inlet'],
    'sensor_3': ['HPC outlet pressure (psia)','High-Pressure Compressor (HPC)'],
    'sensor_4': ['HPT outlet temperature','High-Pressure Turbine (HPT)'],
    'sensor_5': ['LPC outlet temperature','Low-Pressure Compressor (LPC)'],
    'sensor_6': ['Fan speed rpm','Fan'],
    'sensor_7': ['Fan inlet pressure (psia)', 'Overall Inlet'],
    'sensor_8': ['HPC outlet static pressure (psia)','High-Pressure Compressor (HPC)'],
    'sensor_9': ['Corrected fan speed (rpm)','Fan'],
    'sensor_10': ['Ratio of pressure (bypass)Fan', 'Overall'],
    'sensor_11': ['Fuel flow (pph)','Combustion Chamber'],
    'sensor_12': ['LPC outlet temperature (corrected)','Low-Pressure Compressor (LPC)'],
    'sensor_13': ['HPT outlet temperature (corrected)','High-Pressure Turbine (HPT)'],
    'sensor_14': ['Physical fan speed (rpm)','Fan'],
    'sensor_15': ['Physical core engine speed (rpm)','High-Pressure Compressor (HPC) and High-Pressure Turbine (HPT)'],
    'sensor_16': ['Bleed air (units)','Overall Engine System'],
    'sensor_17': ['Fuel-air ratio','Combustion Chamber'],
    'sensor_18': ['Thrust-specific fuel consumption (TSFC)','Overall Engine System'],
    'sensor_19': ['Total temperature at LPC exit','Low-Pressure Compressor (LPC)'],
    'sensor_20': ['Engine pressure ratio','Overall Engine System'],
    'sensor_21': ['Total temperature at HPT exit ','High-Pressure Turbine (HPT)']
}

# ===================== Model Definition =====================
class CNNTransformerModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(cnn_channels, d_model)
        self.fc = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1))

    def forward(self, x):
        x = x.permute(0,2,1)  # [B, F, T]
        x = self.cnn(x)       # [B, C, T]
        x = x.permute(0,2,1)  # [B, T, C]
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ===================== Data Loading & Preprocessing =====================
def load_and_preprocess_data():
    """Load and preprocess training data to fit the scaler"""
    print("Loading training data to fit scaler...")
    train = pd.read_csv(TRAIN_PATH, sep=' ', header=None)
    train.drop([26,27], axis=1, inplace=True)
    cols = ['engine_id','cycle'] + [f'op_setting_{i+1}' for i in range(3)] + [f'sensor_{i+1}' for i in range(21)]
    train.columns = cols
    
    # Calculate RUL
    rul_df = train.groupby('engine_id')['cycle'].max().reset_index()
    rul_df.columns = ['engine_id', 'max_cycle']
    train = train.merge(rul_df, on='engine_id')
    train['RUL'] = train['max_cycle'] - train['cycle']
    train.drop('max_cycle', axis=1, inplace=True)
    train['RUL'] = train['RUL'].clip(upper=MAX_RUL)
    
    # Get feature columns
    feature_cols = train.columns[2:-1].tolist()
    
    # Fit scaler on training data
    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])
    
    return scaler, feature_cols

def load_test_data(scaler, feature_cols):
    """Load and preprocess test data"""
    print("Loading test data...")
    test = pd.read_csv(TEST_PATH, sep=' ', header=None)
    test.drop([26,27], axis=1, inplace=True)
    cols = ['engine_id','cycle'] + [f'op_setting_{i+1}' for i in range(3)] + [f'sensor_{i+1}' for i in range(21)]
    test.columns = cols
    
    # Scale test features
    test[feature_cols] = scaler.transform(test[feature_cols])
    
    return test

def load_true_rul():
    """Load true RUL values"""
    print("Loading true RUL values...")
    truth = pd.read_csv(TRUTH_PATH, header=None)
    truth.columns = ['RUL']
    return truth['RUL'].values

def create_sequences_test_last(df, feature_cols, seq_length=SEQ_LEN):
    """Create sequences from test data (last SEQ_LEN cycles for each engine)"""
    X, engine_ids = [], []
    for eid in df['engine_id'].unique():
        ed = df[df['engine_id']==eid].reset_index(drop=True)
        if len(ed) >= seq_length:
            X.append(ed.iloc[-seq_length:][feature_cols].values)
            engine_ids.append(eid)
        else:
            # If engine has fewer than seq_length cycles, pad with zeros
            seq_data = ed[feature_cols].values
            pad_len = seq_length - len(ed)
            padded = np.vstack([np.zeros((pad_len, len(feature_cols))), seq_data])
            X.append(padded)
            engine_ids.append(eid)
    return np.array(X), np.array(engine_ids)

# ===================== Explainable AI - Feature Attribution =====================
def compute_grad_times_input_smooth_single(model, x_np, n_samples=12, stdev=1e-3):
    """
    Compute feature importance using SmoothGrad (Gradient x Input)
    This explains which sensors/features contributed most to the prediction
    """
    model.eval()
    X = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
    accum = np.zeros_like(x_np)
    
    for _ in range(n_samples):
        noise = torch.randn_like(X) * stdev
        Xn = (X + noise).clone().detach().requires_grad_(True)
        out = model(Xn)
        out.sum().backward()
        grads = Xn.grad.detach().cpu().numpy()[0]
        inputs = Xn.detach().cpu().numpy()[0]
        accum += np.abs(grads * inputs)
    
    return accum / n_samples

def analyze_engine(model, x_test, feature_cols, engine_id, predicted_rul, true_rul):
    """
    Analyze a single engine and return detailed report
    """
    # Compute attribution map
    attr_map = compute_grad_times_input_smooth_single(model, x_test)  # [seq_len, features]
    feat_imp = attr_map.sum(axis=0)  # Sum across time steps
    feat_imp_norm = feat_imp / (feat_imp.sum() + 1e-12)  # Normalize to 0-1
    
    # Get top 5 sensors
    top_idx = feat_imp_norm.argsort()[::-1][:5]
    top_sensors = [(feature_cols[idx], float(feat_imp_norm[idx])) for idx in top_idx]
    
    # Map sensors to subsystems and aggregate importance
    subsystem_scores = {}
    for idx in range(len(feature_cols)):
        sensor_key = feature_cols[idx]
        subs = SENSOR_TO_SUBSYSTEM.get(sensor_key, ['Unknown'])
        for sub in subs:
            subsystem_scores[sub] = subsystem_scores.get(sub, 0.0) + float(feat_imp_norm[idx])
    
    # Sort subsystems by importance
    subs_sorted = sorted(subsystem_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Determine if imminent failure
    imminent = bool(predicted_rul <= THRESHOLD_RUL_FLAG)
    
    return {
        'engine_id': int(engine_id),
        'predicted_RUL': float(predicted_rul),
        'true_RUL': int(true_rul),
        'imminent_failure_flag': imminent,
        'top_sensors': top_sensors,
        'top_subsystems': subs_sorted[:5]
    }

# ===================== Inference =====================
def run_inference():
    """Main inference function with explainable AI"""
    print("="*60)
    print("Engine RUL Prediction - Inference with Explainable AI")
    print("="*60)
    
    # Load scaler and preprocess data
    scaler, feature_cols = load_and_preprocess_data()
    test_df = load_test_data(scaler, feature_cols)
    true_rul_values = load_true_rul()
    
    # Create test sequences
    print("Creating sequences...")
    X_test, test_engine_ids = create_sequences_test_last(test_df, feature_cols, SEQ_LEN)
    print(f"Test sequences shape: {X_test.shape}")
    print(f"Number of test engines: {len(test_engine_ids)}")
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = CNNTransformerModel(input_dim=len(feature_cols), cnn_channels=CNN_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Make predictions
    print("\nMaking predictions...")
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    # Get true RUL for test engines
    true_rul_test = true_rul_values[:len(test_engine_ids)]
    
    # Analyze each engine with feature importance
    print("\nAnalyzing engines with Explainable AI...")
    print("(Computing feature importance for each engine...)")
    reports = []
    
    for i, (eid, pred, true_rul) in enumerate(zip(test_engine_ids, predictions, true_rul_test)):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_engine_ids)} engines...")
        
        report = analyze_engine(model, X_test[i], feature_cols, eid, pred, true_rul)
        reports.append(report)
    
    # Create DataFrame
    results_df = pd.DataFrame(reports)
    
    # Save to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Detailed reports saved to: {OUTPUT_CSV}")
    print(f"[OK] Total engines analyzed: {len(results_df)}")
    
    # Statistics
    imminent_count = results_df['imminent_failure_flag'].sum()
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    print(f"Total engines: {len(results_df)}")
    print(f"Imminent failures (RUL <= {THRESHOLD_RUL_FLAG}): {imminent_count}")
    print(f"Average predicted RUL: {results_df['predicted_RUL'].mean():.2f} cycles")
    print(f"Average true RUL: {results_df['true_RUL'].mean():.2f} cycles")
    
    # Show sample reports
    print(f"\n{'='*60}")
    print("Sample Engine Reports:")
    print(f"{'='*60}")
    for idx in [0, 17, 19]:  # Show engines 1, 18, 20
        if idx < len(results_df):
            row = results_df.iloc[idx]
            print(f"\nEngine ID: {row['engine_id']}")
            print(f"  Predicted RUL: {row['predicted_RUL']:.2f} cycles")
            print(f"  True RUL: {row['true_RUL']} cycles")
            print(f"  Imminent Failure: {'YES - URGENT!' if row['imminent_failure_flag'] else 'No'}")
            print(f"  Top 3 Contributing Sensors:")
            for sensor, score in row['top_sensors'][:3]:
                print(f"    - {sensor}: {score*100:.2f}%")
            print(f"  Top 3 Affected Subsystems:")
            for subsystem, score in row['top_subsystems'][:3]:
                print(f"    - {subsystem}: {score*100:.2f}%")
    
    return results_df

# ===================== Main =====================
if __name__ == "__main__":
    results = run_inference()
    print("\n" + "="*60)
    print("Inference completed successfully!")
    print("="*60)
