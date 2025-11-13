import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report # <-- ADDED THIS IMPORT
import joblib
import lightgbm as lgb
import argparse
import cv2
import serial
import time
from threading import Thread, Lock

# --- 1. NEW: Intelligent Feature Engineering for Seatbelt ---
def create_seatbelt_features(sensor_data):
    """
    Takes an array of 8 sensor values and returns a DataFrame of high-level,
    descriptive features designed to distinguish between recline angles.
    """
    if sensor_data.ndim == 1:
        sensor_data = sensor_data.reshape(1, -1)

    features_list = []
    
    for row in sensor_data:
        features = {}
        epsilon = 1e-6 # To prevent division by zero

        # --- Base Features ---
        for i in range(8):
            features[f'seatbelt_{i}'] = row[i]
        
        # --- Statistical Features ---
        features['mean'] = np.mean(row)
        features['std'] = np.std(row)
        features['max'] = np.max(row)
        features['sum'] = np.sum(row)

        # --- 1D Center of Pressure (CoP) ---
        # This is a powerful feature indicating the center of pressure along the back.
        # A lower value means pressure is higher up the back.
        total_pressure = features['sum']
        if total_pressure > epsilon:
            indices = np.arange(8)
            features['cop_1d'] = np.sum(row * indices) / total_pressure
        else:
            features['cop_1d'] = -1

        # --- Ratio Features ---
        top_sum = np.sum(row[:4]) + epsilon
        bottom_sum = np.sum(row[4:]) + epsilon
        features['top_bottom_ratio'] = top_sum / bottom_sum

        # --- Gradient/Difference Features ---
        # These capture how pressure changes between adjacent sensors.
        for i in range(7):
            features[f'diff_{i}_{i+1}'] = row[i+1] - row[i]
            
        features_list.append(features)

    return pd.DataFrame(features_list)

# --- 2. Training Function ---
def train(args):
    print("--- Starting Model Training with Intelligent Seatbelt Features ---")
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{args.csv_path}'"); return

    # Use only the 8 seatbelt sensors to create features
    seatbelt_cols = [f'sensor_{i}' for i in range(8)]
    X_raw = df[seatbelt_cols].fillna(0).values
    y = df['label_x'].values

    print("Creating intelligent features from the seatbelt data...")
    X_featured = create_seatbelt_features(X_raw)
    print(f"Successfully created {X_featured.shape[1]} features: {X_featured.columns.tolist()}")

    X_train, X_val, y_train, y_val = train_test_split(X_featured, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\nTraining LightGBM model...")
    lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_estimators=200, learning_rate=0.05, num_leaves=20)
    
    # Convert scaled training data back to DataFrame to pass feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_featured.columns)
    
    lgbm.fit(X_train_scaled_df, y_train,
             eval_set=[(X_val_scaled, y_val)],
             eval_metric='multi_logloss',
             callbacks=[lgb.early_stopping(15, verbose=True)])

    # --- UPDATED THIS BLOCK FOR DETAILED EVALUATION ---
    print("\n--- Model Evaluation on Validation Set ---")
    
    # Convert scaled validation data to DataFrame to prevent warning
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_featured.columns)
    y_pred = lgbm.predict(X_val_scaled_df)
    
    target_names = ['130 deg', '110 deg', '95 deg', '85 deg']
    
    # Added digits=4 for four decimal places
    print(classification_report(y_val, y_pred, target_names=target_names, digits=4))
    # --- END OF UPDATED BLOCK ---

    joblib.dump(lgbm, args.model_path)
    joblib.dump(scaler, args.scaler_path)
    print(f"\nTraining complete. Model saved to '{args.model_path}', scaler to '{args.scaler_path}'.")


# --- 3. Advanced Filtering & Animation Classes (Unchanged) ---
class HoltSmoother:
    def __init__(self, num_classes, alpha=0.1, beta=0.08):
        self.alpha, self.beta, self.level, self.trend = alpha, beta, np.ones(num_classes)/num_classes, np.zeros(num_classes)
    def add(self, p):
        last_level = self.level
        self.level = self.alpha*p + (1-self.alpha)*(last_level+self.trend)
        self.trend = self.beta*(self.level-last_level) + (1-self.beta)*self.trend
    def get_stable_prediction(self): return np.argmax(self.level + self.trend)

class EaseInOutStickman:
    def __init__(self, templates, animation_duration=25):
        self.templates, self.duration, self.frame, self.animating = templates, animation_duration, 0, False
        self.start_lm, self.current_lm, self.target_lm = templates[1].copy(), templates[1].copy(), templates[1].copy()
    def _ease(self, t): t*=2; return 0.5*t*t*t if t<1 else 0.5*((t-2)*(t-2)*(t-2)+2)
    def set_target(self, label):
        if label in self.templates and self.target_lm != self.templates[label]:
            self.target_lm, self.start_lm, self.frame, self.animating = self.templates[label], self.current_lm.copy(), 0, True
    def update(self):
        if not self.animating: return
        self.frame += 1
        progress = self._ease(min(1.0, self.frame / self.duration))
        for k in self.current_lm: self.current_lm[k] = (self.start_lm[k][0]+(self.target_lm[k][0]-self.start_lm[k][0])*progress, self.start_lm[k][1]+(self.target_lm[k][1]-self.start_lm[k][1])*progress)
        if self.frame >= self.duration: self.animating, self.current_lm = False, self.target_lm.copy()
    def draw(self, frame, name):
        lm = {k: (int(v[0]), int(v[1])) for k, v in self.current_lm.items()}
        for p1, p2 in [('LE','LS'),('LS','LH'),('LK','LH')]:
            cv2.line(frame,lm[p1],lm[p2],(0,255,0),4); cv2.circle(frame,lm[p1],10,(0,255,255),-1); cv2.circle(frame,lm[p2],10,(0,255,255),-1)
        cv2.putText(frame,f"Posture: {name}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3,cv2.LINE_AA)
        return frame

# --- 4. Reconstruction Function ---
def reconstruct(args):
    print("--- Starting Real-Time Reconstruction with Intelligent Features ---")
    try:
        model, scaler, posture_templates = (joblib.load(args.model_path), joblib.load(args.scaler_path), get_posture_templates(args.csv_path))
        print("Model, scaler, and templates loaded successfully.")
    except Exception as e:
        print(f"Error loading files: {e}\nDid you run training first with 'python {__file__} --mode train'?")
        return

    smoother, stickman = HoltSmoother(num_classes=4), EaseInOutStickman(posture_templates)
    label_to_angle = {0: '130 deg', 1: '110 deg', 2: '95 deg', 3: '85 deg'}

    try:
        ser_eight = serial.Serial(args.eight_port, 115200, timeout=1)
        print(f"Connected to seatbelt sensor on {args.eight_port}")
    except serial.SerialException as e:
        print(f"Error opening serial port {args.eight_port}: {e}"); return

    latest_eight_data, lock, stop_thread = [0] * 8, Lock(), False
    def read_sensor():
        nonlocal latest_eight_data
        while not stop_thread:
            try:
                line = ser_eight.readline().decode('utf-8').strip()
                if line and len(parts := [int(p) for p in line.split(',') if p.strip().isdigit()]) == 8:
                    with lock: latest_eight_data = parts
            except: continue
    Thread(target=read_sensor, daemon=True).start()

    print("\nStarting reconstruction loop. Press 'ESC' in the window to exit.")
    try:
        while True:
            canvas = np.zeros((480, 640, 3), dtype="uint8")
            with lock: live_raw_data = np.array(latest_eight_data)
            
            live_featured = create_seatbelt_features(live_raw_data)
            live_scaled_np = scaler.transform(live_featured)
            live_scaled_df = pd.DataFrame(live_scaled_np, columns=live_featured.columns)
            
            probabilities = model.predict_proba(live_scaled_df)[0]
            smoother.add(probabilities)
            
            stable_prediction = smoother.get_stable_prediction()
            stickman.set_target(stable_prediction)
            stickman.update()
            
            posture_name = label_to_angle.get(stable_prediction, "Unknown")
            canvas = stickman.draw(canvas, posture_name)
            
            print(f"Probs -> 130°:{probabilities[0]:.2f} | 110°:{probabilities[1]:.2f} | 95°:{probabilities[2]:.2f} | 85°:{probabilities[3]:.2f}", end='\r')
            cv2.imshow("Posture Reconstruction", canvas)
            if cv2.waitKey(20) & 0xFF == 27: break
    finally:
        stop_thread = True; ser_eight.close(); cv2.destroyAllWindows(); print("\nApplication closed.")

def get_posture_templates(csv_path):
    try:
        df = pd.read_csv(csv_path)
        templates = {}
        for label, group in df.groupby('label_x'):
            templates[label] = { 'LS': (int(group['LS_x'].mean()), int(group['LS_y'].mean())), 'LE': (int(group['LE_x'].mean()), int(group['LE_y'].mean())), 'LH': (int(group['LH_x'].mean()), int(group['LH_y'].mean())), 'LK': (int(group['LK_x'].mean()), int(group['LK_y'].mean())) }
        return templates
    except FileNotFoundError: return None

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A robust posture recognition application using intelligent features from seatbelt sensors.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'recon'])
    parser.add_argument('--csv-path', type=str, default='siddhant.csv')
    parser.add_argument('--eight-port', type=str, help="[Recon mode] COM port for the 8-channel seatbelt sensor.")
    parser.add_argument('--model-path', type=str, default='seatbelt_features_model.joblib')
    parser.add_argument('--scaler-path', type=str, default='seatbelt_features_scaler.joblib')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recon':
        if not args.eight_port:
            parser.error("--eight-port is required for 'recon' mode.")
        reconstruct(args)
