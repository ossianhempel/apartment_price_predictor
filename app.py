from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

# Route for a prediction page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('prediction_form.html')
    else:
        data = CustomData(
            number_of_rooms = int(request.form['number_of_rooms']),
            area_size = int(request.form['area_size']),
            year_built = int(request.form['year_built']),
            annual_fee_sek = int(request.form['annual_fee_sek']),
            region = request.form['region'],
            has_balcony = request.form.get('has_balcony', 'no'),
            floor_number = int(request.form['floor_number']),
        )

        prediction_df = data.get_data_as_datafrane()
        print(prediction_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(prediction_df)
        formatted_result = round(result[0] / 1_000_000, 2)  # Formats the result in millions with 2 decimal places

        return render_template('prediction_form.html', result=f"{formatted_result} million SEK")

    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True) # for dev purposes
    # http://127.0.0.1:3000/predict