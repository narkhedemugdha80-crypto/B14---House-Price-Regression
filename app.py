import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained linear regression model
with open('linear.pkl', 'rb') as f:
    model = pickle.load(f)

# Embedded HTML and CSS layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        :root {
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f1f5f9;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #cbd5e1;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 650px;
        }
        h1 {
            text-align: center;
            color: var(--primary-color);
            margin-bottom: 8px;
            font-size: 28px;
        }
        p.subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            margin-bottom: 5px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 13px;
            color: #475569;
        }
        input {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        button {
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 20px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 25px;
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .result-card h2 {
            margin: 0 0 5px 0;
            color: #312e81;
            font-size: 16px;
        }
        .prediction-value {
            font-size: 32px;
            font-weight: bold;
            color: var(--primary-color);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>House Price Predictor</h1>
        <p class="subtitle">Enter the property attributes below to evaluate the estimated market value</p>
        
        <form action="/predict" method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Square_Footage">Square Footage</label>
                    <input type="number" step="any" id="Square_Footage" name="Square_Footage" required placeholder="e.g. 2100">
                </div>
                <div class="form-group">
                    <label for="Num_Bedrooms">Number of Bedrooms</label>
                    <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" required placeholder="e.g. 3">
                </div>
                <div class="form-group">
                    <label for="Num_Bathrooms">Number of Bathrooms</label>
                    <input type="number" step="any" id="Num_Bathrooms" name="Num_Bathrooms" required placeholder="e.g. 2">
                </div>
                <div class="form-group">
                    <label for="Year_Built">Year Built</label>
                    <input type="number" id="Year_Built" name="Year_Built" required placeholder="e.g. 2012">
                </div>
                <div class="form-group">
                    <label for="Lot_Size">Lot Size</label>
                    <input type="number" step="any" id="Lot_Size" name="Lot_Size" required placeholder="e.g. 4500">
                </div>
                <div class="form-group">
                    <label for="Garage_Size">Garage Size</label>
                    <input type="number" id="Garage_Size" name="Garage_Size" required placeholder="e.g. 2">
                </div>
                <div class="form-group full-width">
                    <label for="Neighborhood_Quality">Neighborhood Quality</label>
                    <input type="number" step="any" id="Neighborhood_Quality" name="Neighborhood_Quality" required placeholder="e.g. 7">
                </div>
            </div>
            <button type="submit">Predict Price</button>
        </form>

        {% if prediction_text %}
        <div class="result-card">
            <h2>Estimated Property Price</h2>
            <div class="prediction-value">{{ prediction_text }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract fields matching the exact order from the training model[cite: 1]
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        final_features = np.array([features])
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        formatted_output = f"${output:,.2f}"
        return render_template_string(HTML_TEMPLATE, prediction_text=formatted_output)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
