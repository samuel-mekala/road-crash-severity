import os
from flask import Flask, render_template, request, jsonify
from predict import predictor, FEATURE_MAPPINGS

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', mappings=FEATURE_MAPPINGS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        result = predictor.predict(data)
        
        if request.is_json:
            return jsonify({'status': 'success', 'data': result})
        
        return render_template('index.html', mappings=FEATURE_MAPPINGS, result=result, input_data=data)
    except Exception as e:
        error_msg = f"Error processing prediction: {str(e)}"
        if request.is_json:
            return jsonify({'status': 'error', 'message': error_msg}), 400
        return render_template('index.html', mappings=FEATURE_MAPPINGS, error=error_msg)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_initialized': predictor.initialized})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True)
