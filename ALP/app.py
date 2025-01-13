from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and preprocessing components
best_rf = joblib.load('best_rf_model.pkl')
obesity_encoder = joblib.load('NObeyesdad_label_encoder.pkl')
columns = joblib.load('columns.pkl')

nobeyesdad_mapping = {
    0: 'Insufficient Weight',
    1: 'Normal Weight',
    2: 'Obesity Type 1',
    3: 'Obesity Type 2',
    4: 'Obesity Type 3',
    5: 'Overweight Level 1',
    6: 'Overweight Level 2'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        input_data = request.json
        print("Input Data Received from Web Form:", input_data)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Log the columns of the DataFrame
        print("Columns in Input DataFrame:", input_df.columns)

        # Ensure all required columns are present
        for col in columns:
            if col not in input_df.columns:
                print(f"Missing column: {col}")
                input_df[col] = 0  # Add missing columns with default value

        # Reorder columns to match model input
        input_df = input_df[columns]
        print("Transformed Input DataFrame:", input_df)

        # Make prediction
        prediction = best_rf.predict(input_df)[0]
        prediction_label = nobeyesdad_mapping.get(prediction, "Unknown")

        return jsonify({'prediction': prediction_label})
    except Exception as e:
        print("Error in Prediction:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
