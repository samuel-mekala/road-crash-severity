import unittest
from app import app
from predict import predictor

class TestRoadCrashApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Road Crash Injury Severity Prediction', response.data)

    def test_prediction_engine(self):
        sample_input = {
            'latitude': 51.5074,
            'longitude': -0.1278,
            'road_number': 10,
            'day_of_week': 2,
            'number_of_vehicles': 3,
            'number_of_casualties': 2,
            'speed_limit': 60,
            'weather_conditions': 2,
            'light_conditions': 4,
            'road_surface_conditions': 2,
            'urban_or_rural_area': 2
        }
        res = predictor.predict(sample_input)
        self.assertIn('severity_label', res)
        self.assertIn(res['severity_label'], ['Slight', 'Serious', 'Fatal'])
        self.assertIn('risk_score', res)
        self.assertTrue(len(res['recommendations']) > 0)

    def test_predict_api_endpoint(self):
        payload = {
            'latitude': 51.5074,
            'longitude': -0.1278,
            'road_number': 10,
            'day_of_week': 2,
            'number_of_vehicles': 2,
            'number_of_casualties': 1,
            'speed_limit': 30,
            'weather_conditions': 1,
            'light_conditions': 1,
            'road_surface_conditions': 1,
            'urban_or_rural_area': 1
        }
        response = self.app.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('severity_label', data['data'])

if __name__ == '__main__':
    unittest.main()
