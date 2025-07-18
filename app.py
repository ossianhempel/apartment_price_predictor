from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

# route for a prediction page
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST']) # change back to route /predict if you want to have another index page
def predict():
    if request.method == 'GET':
        return render_template('prediction_form.html')
    else:
        # Extract form data
        form_data = {
            'number_of_rooms': request.form['number_of_rooms'],
            'area_size': request.form['area_size'],
            'year_built': request.form['year_built'],
            'annual_fee_sek': request.form['annual_fee_sek'],
            'region': request.form['region'],
            'has_balcony': request.form.get('has_balcony', 'no'),
            'floor_number': request.form['floor_number'],
        }
        
        data = CustomData(
            number_of_rooms = int(form_data['number_of_rooms']),
            area_size = int(form_data['area_size']),
            year_built = int(form_data['year_built']),
            annual_fee_sek = int(form_data['annual_fee_sek']),
            region = form_data['region'],
            has_balcony = form_data['has_balcony'],
            floor_number = int(form_data['floor_number']),
        )

        prediction_df = data.get_data_as_datafrane()
        print(prediction_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(prediction_df)
        formatted_result = round(result[0] / 1_000_000, 2)  # Formats the result in millions with 2 decimal places

        return render_template('prediction_form.html', 
                             result=f"{formatted_result} million SEK",
                             form_data=form_data)

    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True) # for dev purposes
    # http://127.0.0.1:3000/predict