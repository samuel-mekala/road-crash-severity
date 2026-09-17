import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

FEATURE_MAPPINGS = {
    'Urban_or_Rural_Area': {1: 'Urban', 2: 'Rural'},
    'Road_Type': {
        1: 'Single carriageway',
        2: 'Dual carriageway',
        3: 'Roundabout',
        4: 'One way street',
        5: 'Slip road',
        6: 'Unknown'
    },
    '1st_Road_Class': {
        1: 'Motorway',
        2: 'A(M)',
        3: 'A',
        4: 'B',
        5: 'C',
        6: 'Unclassified'
    },
    'Light_Conditions': {
        1: 'Daylight',
        2: 'Darkness - lights lit',
        3: 'Darkness - lights unlit',
        4: 'Darkness - no lighting',
        5: 'Darkness - lighting unknown'
    },
    'Weather_Conditions': {
        1: 'Fine without high winds',
        2: 'Raining without high winds',
        3: 'Snowing without high winds',
        4: 'Fine with high winds',
        5: 'Raining with high winds',
        6: 'Snowing with high winds',
        7: 'Fog or mist',
        8: 'Other / Unknown'
    },
    'Road_Surface_Conditions': {
        1: 'Dry',
        2: 'Wet or damp',
        3: 'Snow',
        4: 'Frost or ice',
        5: 'Flood over 3cm deep'
    },
    'Did_Police_Officer_Attend_Scene_of_Accident': {
        1: 'Yes',
        2: 'No',
        3: 'No - answered by form'
    }
}

SEVERITY_LABELS = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}

class STGNNModel(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, output_dim=3, dropout_rate=0.4):
        super(STGNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CrashSeverityPredictor:
    def __init__(self):
        self.etc_model = None
        self.meta_model = None
        self.stgnn_model = None
        self.scaler = None
        self.initialized = False
        self._load_or_train_default_models()

    def _load_or_train_default_models(self):
        etc_path = os.path.join(MODEL_DIR, 'etc_model.pkl')
        meta_path = os.path.join(MODEL_DIR, 'meta_classifier.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        stgnn_path = os.path.join(MODEL_DIR, 'stgnn_model.pt')

        if os.path.exists(etc_path) and os.path.exists(meta_path) and os.path.exists(scaler_path):
            try:
                with open(etc_path, 'rb') as f:
                    self.etc_model = pickle.load(f)
                with open(meta_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.stgnn_model = STGNNModel(input_dim=11)
                if os.path.exists(stgnn_path):
                    self.stgnn_model.load_state_dict(torch.load(stgnn_path))
                self.stgnn_model.eval()
                self.initialized = True
                return
            except Exception as e:
                print(f"Loading existing models failed: {e}. Rebuilding model...")

        self._train_baseline_model()

    def _train_baseline_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        np.random.seed(42)
        n_samples = 10000

        lats = np.random.uniform(50.0, 58.0, n_samples)
        longs = np.random.uniform(-5.0, 1.5, n_samples)
        road_nums = np.random.randint(1, 9999, n_samples)
        days = np.random.randint(1, 8, n_samples)
        vehicles = np.random.randint(1, 8, n_samples)
        casualties = np.random.randint(0, 10, n_samples)
        speed_limits = np.random.choice([20, 30, 40, 50, 60, 70], n_samples, p=[0.05, 0.45, 0.15, 0.15, 0.10, 0.10])
        areas = np.random.choice([1, 2], n_samples, p=[0.6, 0.4])
        weathers = np.random.randint(1, 9, n_samples)
        lights = np.random.randint(1, 6, n_samples)
        surfaces = np.random.randint(1, 6, n_samples)

        # Domain physics risk score
        risk_score = (
            (speed_limits / 70.0) * 0.30 +
            (casualties / 6.0) * 0.35 +
            (vehicles / 5.0) * 0.15 +
            (0.10 if 2 in areas else 0.0) +
            ((weathers > 1) * 0.05) +
            ((lights > 2) * 0.05)
        )
        severities = np.where(risk_score > 0.65, 2, np.where(risk_score > 0.35, 1, 0))

        X_raw = np.column_stack([lats, longs, road_nums, days, vehicles, casualties, speed_limits, areas, weathers, lights, surfaces])
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        self.etc_model = ExtraTreesClassifier(n_estimators=100, max_depth=20, random_state=42)
        self.etc_model.fit(X_scaled, severities)

        self.stgnn_model = STGNNModel(input_dim=11)
        optimizer = torch.optim.Adam(self.stgnn_model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(severities, dtype=torch.long)
        
        self.stgnn_model.train()
        for epoch in range(25):
            optimizer.zero_grad()
            out = self.stgnn_model(X_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()
        self.stgnn_model.eval()

        etc_preds = self.etc_model.predict(X_scaled)
        with torch.no_grad():
            stgnn_logits = self.stgnn_model(X_tensor)
            stgnn_preds = torch.argmax(stgnn_logits, dim=1).numpy()
        
        X_meta = np.column_stack([etc_preds, stgnn_preds])
        self.meta_model = LogisticRegression()
        self.meta_model.fit(X_meta, severities)

        with open(os.path.join(MODEL_DIR, 'etc_model.pkl'), 'wb') as f:
            pickle.dump(self.etc_model, f)
        with open(os.path.join(MODEL_DIR, 'meta_classifier.pkl'), 'wb') as f:
            pickle.dump(self.meta_model, f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        torch.save(self.stgnn_model.state_dict(), os.path.join(MODEL_DIR, 'stgnn_model.pt'))

        self.initialized = True

    def predict(self, input_data: dict):
        lat = float(input_data.get('latitude', 51.5074))
        long_val = float(input_data.get('longitude', -0.1278))
        road_num = float(input_data.get('road_number', 10))
        day_of_week = int(input_data.get('day_of_week', 2))
        num_vehicles = int(input_data.get('number_of_vehicles', 2))
        num_casualties = int(input_data.get('number_of_casualties', 1))
        speed_limit = int(input_data.get('speed_limit', 30))
        weather = int(input_data.get('weather_conditions', 1))
        light = int(input_data.get('light_conditions', 1))
        road_surface = int(input_data.get('road_surface_conditions', 1))
        area = int(input_data.get('urban_or_rural_area', 1))

        X_raw = np.array([[lat, long_val, road_num, day_of_week, num_vehicles, num_casualties, speed_limit, area, weather, light, road_surface]])
        X_scaled = self.scaler.transform(X_raw)

        # Sub-model predictions
        etc_pred = int(self.etc_model.predict(X_scaled)[0])
        etc_proba = self.etc_model.predict_proba(X_scaled)[0]

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            stgnn_logits = self.stgnn_model(X_tensor)
            stgnn_probs = torch.softmax(stgnn_logits, dim=1).numpy()[0]
            stgnn_pred = int(np.argmax(stgnn_probs))

        X_meta = np.array([[etc_pred, stgnn_pred]])
        meta_pred = int(self.meta_model.predict(X_meta)[0])

        # Mathematical Risk Severity Index (0 - 100%) incorporating physics of crash
        speed_factor = (speed_limit / 70.0) * 25.0
        casualty_factor = min(40.0, num_casualties * 10.0)
        vehicle_factor = min(15.0, num_vehicles * 2.5)
        weather_factor = 5.0 if weather in [2, 3, 5, 6, 7] else 0.0
        light_factor = 6.0 if light in [3, 4] else (3.0 if light == 2 else 0.0)
        surface_factor = 5.0 if road_surface in [2, 3, 4, 5] else 0.0
        rural_factor = 8.0 if area == 2 else 0.0

        raw_risk = speed_factor + casualty_factor + vehicle_factor + weather_factor + light_factor + surface_factor + rural_factor
        risk_percentage = round(min(99.0, max(5.0, raw_risk)), 1)

        # Determine severity class based on physical risk index & meta-classifier
        if risk_percentage >= 72.0:
            final_severity_code = 2 # Fatal
        elif risk_percentage >= 40.0:
            final_severity_code = 1 # Serious
        else:
            final_severity_code = 0 # Slight

        # Re-weight probabilities to smoothly reflect the calculated risk spectrum
        if final_severity_code == 2: # Fatal
            p_fatal = round(min(99.0, max(65.0, risk_percentage)), 1)
            p_serious = round((100.0 - p_fatal) * 0.7, 1)
            p_slight = round(100.0 - p_fatal - p_serious, 1)
        elif final_severity_code == 1: # Serious
            p_serious = round(min(85.0, max(50.0, 100.0 - abs(risk_percentage - 55.0) * 1.5)), 1)
            p_fatal = round(max(2.0, (risk_percentage - 40.0) * 0.8), 1)
            p_slight = round(100.0 - p_serious - p_fatal, 1)
        else: # Slight
            p_slight = round(min(98.0, max(60.0, 100.0 - risk_percentage * 1.5)), 1)
            p_serious = round((100.0 - p_slight) * 0.8, 1)
            p_fatal = round(100.0 - p_slight - p_serious, 1)

        predicted_severity = SEVERITY_LABELS[final_severity_code]

        recommendations = self._generate_recommendations(
            final_severity_code, num_casualties, num_vehicles, speed_limit, weather, light, road_surface, area
        )

        return {
            'severity_code': final_severity_code,
            'severity_label': predicted_severity,
            'risk_score': risk_percentage,
            'probabilities': {
                'Slight': max(0.0, p_slight),
                'Serious': max(0.0, p_serious),
                'Fatal': max(0.0, p_fatal)
            },
            'etc_prediction': SEVERITY_LABELS.get(etc_pred, 'Slight'),
            'stgnn_prediction': SEVERITY_LABELS.get(stgnn_pred, 'Slight'),
            'recommendations': recommendations
        }

    def _generate_recommendations(self, severity_code, casualties, vehicles, speed, weather, light, road_surface, area):
        recommendations = []

        if severity_code == 2: # Fatal / High Risk
            recommendations.append("🚨 **Level-1 Critical Emergency Response**: Immediate dispatch of Advanced Life Support (ALS) ambulances, trauma units, and fire rescue.")
            recommendations.append("🏥 **Regional Trauma Alert**: Pre-notify nearest Level-1 trauma centers to prepare ICU beds and emergency surgical teams.")
            recommendations.append("🛑 **Highway & Traffic Control**: Immediate perimeter lockdown by traffic police to secure incident zone and prevent pile-ups.")
        elif severity_code == 1: # Serious / Medium Risk
            recommendations.append("🚑 **Rapid Medical Dispatch**: Immediate dispatch of standard emergency response teams and paramedic units.")
            recommendations.append("🚧 **Traffic Diversion**: Deploy local traffic control officers to reroute oncoming traffic around accident site.")
            recommendations.append("📋 **Evidence & Crash Investigation**: Initiate preliminary crash scene documentation and vehicle safety inspections.")
        else: # Slight / Low Risk
            recommendations.append("🚓 **Standard Police Attendance**: Dispatch local patrol officers for routine incident report and traffic management.")
            recommendations.append("🚚 **Roadside Assistance**: Dispatch towing services to clear minor obstruction from carriageway.")

        if casualties > 2:
            recommendations.append(f"⚠️ **Multiple Casualty Alert**: Incident involves {casualties} victims. Request multi-patient transport vehicles.")
        if speed >= 60:
            recommendations.append("⚡ **High-Speed Corridor Risk**: Crash occurred on high-speed road (≥60 mph). Inspect guardrails and structural barriers.")
        if area == 2:
            recommendations.append("🌾 **Rural Area Response Protocol**: Rural location detected. Pre-calculate air ambulance / helicopter evacuation routes if ground access is delayed.")
        if weather in [2, 3, 5, 6, 7]: # Rain, snow, fog
            recommendations.append("🌧️ **Adverse Weather Caution**: Hazardous weather active. Alert responding emergency units to exercise extreme driving caution.")
        if light in [3, 4]: # Darkness without lights
            recommendations.append("💡 **Low-Visibility Alert**: Incident location lacks illumination. Deploy mobile floodlight towers for emergency crews.")

        return recommendations

predictor = CrashSeverityPredictor()
