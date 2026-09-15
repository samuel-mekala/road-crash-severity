import requests
import json

BASE_URL = "http://127.0.0.1:5050"

scenarios = [
    {
        "name": "Scenario A: Minor Urban Incident",
        "payload": {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "road_number": 10,
            "day_of_week": 2,
            "number_of_vehicles": 1,
            "number_of_casualties": 1,
            "speed_limit": 20,
            "weather_conditions": 1, # Fine without high winds
            "light_conditions": 1,   # Daylight
            "road_surface_conditions": 1, # Dry
            "urban_or_rural_area": 1 # Urban
        }
    },
    {
        "name": "Scenario B: Moderate Rural Collision",
        "payload": {
            "latitude": 52.2053,
            "longitude": -0.1218,
            "road_number": 45,
            "day_of_week": 5,
            "number_of_vehicles": 2,
            "number_of_casualties": 2,
            "speed_limit": 40,
            "weather_conditions": 2, # Rain
            "light_conditions": 2,   # Darkness lit
            "road_surface_conditions": 2, # Wet
            "urban_or_rural_area": 2 # Rural
        }
    },
    {
        "name": "Scenario C: Severe Highway Pile-up",
        "payload": {
            "latitude": 54.5973,
            "longitude": -3.1883,
            "road_number": 1,
            "day_of_week": 6,
            "number_of_vehicles": 5,
            "number_of_casualties": 6,
            "speed_limit": 70,
            "weather_conditions": 5, # Raining + High winds
            "light_conditions": 4,   # Darkness no lights
            "road_surface_conditions": 2, # Wet
            "urban_or_rural_area": 2 # Rural
        }
    }
]

print("==========================================================")
print("END-TO-END MODEL BEHAVIOR & INFERENCE VERIFICATION TEST")
print("==========================================================")

for s in scenarios:
    print(f"\n--- {s['name']} ---")
    res = requests.post(f"{BASE_URL}/predict", json=s['payload'])
    data = res.json()['data']
    
    print(f"Final Ensemble Prediction : {data['severity_label']} (Code {data['severity_code']})")
    print(f"ExtraTrees Prediction     : {data['etc_prediction']}")
    print(f"ST-GNN Prediction         : {data['stgnn_prediction']}")
    print(f"Risk Score (%)            : {data['risk_score']}%")
    print(f"Probabilities Breakdown   : Slight: {data['probabilities']['Slight']}%, Serious: {data['probabilities']['Serious']}%, Fatal: {data['probabilities']['Fatal']}%")
    print("Emergency Recommendations:")
    for rec in data['recommendations']:
        print(f"  - {rec}")

print("\n==========================================================")
print("VERIFICATION COMPLETE")
print("==========================================================")
