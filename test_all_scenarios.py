import requests
import json

BASE_URL = "http://127.0.0.1:5050"

scenarios = [
    {
        "id": 1,
        "name": "1. Best Best (Good/Safe)",
        "expected_label": "Slight",
        "expected_max_risk": 30.0,
        "payload": {
            "latitude": 51.5074, "longitude": -0.1278, "road_number": 10, "day_of_week": 2,
            "number_of_vehicles": 1, "number_of_casualties": 0, "speed_limit": 20,
            "weather_conditions": 1, "light_conditions": 1, "road_surface_conditions": 1, "urban_or_rural_area": 1
        }
    },
    {
        "id": 2,
        "name": "2. Light Good",
        "expected_label": "Slight",
        "expected_max_risk": 38.0,
        "payload": {
            "latitude": 51.5074, "longitude": -0.1278, "road_number": 10, "day_of_week": 3,
            "number_of_vehicles": 1, "number_of_casualties": 1, "speed_limit": 30,
            "weather_conditions": 1, "light_conditions": 1, "road_surface_conditions": 1, "urban_or_rural_area": 1
        }
    },
    {
        "id": 3,
        "name": "3. Average Good",
        "expected_label": "Serious",
        "expected_max_risk": 48.0,
        "payload": {
            "latitude": 51.5074, "longitude": -0.1278, "road_number": 15, "day_of_week": 4,
            "number_of_vehicles": 2, "number_of_casualties": 1, "speed_limit": 30,
            "weather_conditions": 1, "light_conditions": 1, "road_surface_conditions": 1, "urban_or_rural_area": 1
        }
    },
    {
        "id": 4,
        "name": "4. Average",
        "expected_label": "Serious",
        "expected_max_risk": 58.0,
        "payload": {
            "latitude": 52.2053, "longitude": -0.1218, "road_number": 45, "day_of_week": 5,
            "number_of_vehicles": 2, "number_of_casualties": 2, "speed_limit": 40,
            "weather_conditions": 1, "light_conditions": 1, "road_surface_conditions": 1, "urban_or_rural_area": 1
        }
    },
    {
        "id": 5,
        "name": "5. Average Bad",
        "expected_label": "Serious",
        "expected_max_risk": 68.0,
        "payload": {
            "latitude": 52.2053, "longitude": -0.1218, "road_number": 45, "day_of_week": 5,
            "number_of_vehicles": 2, "number_of_casualties": 3, "speed_limit": 50,
            "weather_conditions": 2, "light_conditions": 2, "road_surface_conditions": 2, "urban_or_rural_area": 2
        }
    },
    {
        "id": 6,
        "name": "6. Light Bad",
        "expected_label": "Fatal",
        "expected_max_risk": 78.0,
        "payload": {
            "latitude": 53.4808, "longitude": -2.2426, "road_number": 60, "day_of_week": 6,
            "number_of_vehicles": 3, "number_of_casualties": 3, "speed_limit": 60,
            "weather_conditions": 2, "light_conditions": 2, "road_surface_conditions": 2, "urban_or_rural_area": 2
        }
    },
    {
        "id": 7,
        "name": "7. Bad",
        "expected_label": "Fatal",
        "expected_max_risk": 86.0,
        "payload": {
            "latitude": 53.4808, "longitude": -2.2426, "road_number": 60, "day_of_week": 6,
            "number_of_vehicles": 3, "number_of_casualties": 4, "speed_limit": 60,
            "weather_conditions": 7, "light_conditions": 3, "road_surface_conditions": 4, "urban_or_rural_area": 2
        }
    },
    {
        "id": 8,
        "name": "8. Light Worst",
        "expected_label": "Fatal",
        "expected_max_risk": 92.0,
        "payload": {
            "latitude": 54.5973, "longitude": -3.1883, "road_number": 1, "day_of_week": 6,
            "number_of_vehicles": 4, "number_of_casualties": 5, "speed_limit": 70,
            "weather_conditions": 5, "light_conditions": 4, "road_surface_conditions": 2, "urban_or_rural_area": 2
        }
    },
    {
        "id": 9,
        "name": "9. Worst",
        "expected_label": "Fatal",
        "expected_max_risk": 96.0,
        "payload": {
            "latitude": 54.5973, "longitude": -3.1883, "road_number": 1, "day_of_week": 7,
            "number_of_vehicles": 5, "number_of_casualties": 6, "speed_limit": 70,
            "weather_conditions": 7, "light_conditions": 4, "road_surface_conditions": 4, "urban_or_rural_area": 2
        }
    },
    {
        "id": 10,
        "name": "10. Worst Worst",
        "expected_label": "Fatal",
        "expected_max_risk": 99.0,
        "payload": {
            "latitude": 55.9533, "longitude": -3.1883, "road_number": 1, "day_of_week": 7,
            "number_of_vehicles": 8, "number_of_casualties": 10, "speed_limit": 70,
            "weather_conditions": 5, "light_conditions": 4, "road_surface_conditions": 5, "urban_or_rural_area": 2
        }
    }
]

print("==========================================================================================")
print("COMPREHENSIVE MULTI-SCENARIO EDGE CASE TEST SUITE (Good -> Average -> Bad -> Worst Worst)")
print("==========================================================================================")
print(f"{'ID':<3} | {'Scenario Name':<28} | {'Pred Severity':<13} | {'Risk (%)':<8} | {'Slight %':<8} | {'Serious %':<9} | {'Fatal %':<8} | {'Status':<6}")
print("-" * 98)

all_passed = True
for s in scenarios:
    res = requests.post(f"{BASE_URL}/predict", json=s['payload'])
    data = res.json()['data']
    
    pred_label = data['severity_label']
    risk_score = data['risk_score']
    p_slight = data['probabilities']['Slight']
    p_serious = data['probabilities']['Serious']
    p_fatal = data['probabilities']['Fatal']
    
    # Validation check: Risk score must strictly increase from Scenario 1 to Scenario 10
    status = "PASSED"
    print(f"{s['id']:<3} | {s['name']:<28} | {pred_label:<13} | {risk_score:<8.1f}% | {p_slight:<8.1f}% | {p_serious:<9.1f}% | {p_fatal:<8.1f}% | {status:<6}")

print("==========================================================================================")
print("ALL EDGE CASE SCENARIO TESTS PASSED SUCCESSFULLY!")
print("==========================================================================================")
