from flask import Flask, request, render_template
import os
import sys
import numpy as np 
import pandas as pd
import pickle

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html') 

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        
        # Total_Stops mapping
        stops_mapping = {
            'Non-stop': 0,
            '1 Stop': 1,
            '2 Stops': 2,
            '3 Stops': 3
        }
        
        source_city = request.form.get('Source')
        dest_city = request.form.get('Destination')
        total_stops_str = request.form.get('Total_Stops')
        
        # Add prefixes like in training
        source_with_prefix = "S_" + source_city
        dest_with_prefix = "D_" + dest_city
        
        # Map Total_Stops to numerical value
        total_stops_numeric = stops_mapping.get(total_stops_str, 0)

        data = CustomData(
            Airline=request.form.get('Airline'),
            Source=source_with_prefix,
            Destination=dest_with_prefix,
            Total_Stops=total_stops_numeric,
            dept_day=int(request.form.get('dept_day')),
            dept_month=int(request.form.get('dept_month')),
            dept_hour=int(request.form.get('dept_hour')),
            dept_minute=int(request.form.get('dept_minute')),
            arrival_day=int(request.form.get('arrival_day')),
            arrival_month=int(request.form.get('arrival_month')),
            arrival_hour=int(request.form.get('arrival_hour')),
            arrival_minute=int(request.form.get('arrival_minute')),
            duration_hours=int(request.form.get('duration_hours')),
            duration_minutes=int(request.form.get('duration_minutes')),
            Route1=request.form.get('Route1'),
            Route2=request.form.get('Route2'),
            Route3=request.form.get('Route3'),
            Route4=request.form.get('Route4'),
            Route5=request.form.get('Route5')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        
        predicted_price = results[0]
        # Original price is in Indian Rupees
        result_text = f"₹{predicted_price:,.2f}"
        # Convert to US Dollars (assuming 1 Rupee = 0.011 USD)
        us_dollar = f"${predicted_price*0.011:,.2f}"

        # Pass form values back to template to preserve them
        form_values = {
            'Airline': request.form.get('Airline'),
            'Source': source_city,
            'Destination': dest_city,
            'Total_Stops': total_stops_str,
            'dept_day': request.form.get('dept_day'),
            'dept_month': request.form.get('dept_month'),
            'dept_hour': request.form.get('dept_hour'),
            'dept_minute': request.form.get('dept_minute'),
            'duration_hours': request.form.get('duration_hours'),
            'duration_minutes': request.form.get('duration_minutes'),
            'arrival_day': request.form.get('arrival_day'),
            'arrival_month': request.form.get('arrival_month'),
            'arrival_hour': request.form.get('arrival_hour'),
            'arrival_minute': request.form.get('arrival_minute'),
            'Route1': request.form.get('Route1'),
            'Route2': request.form.get('Route2'),
            'Route3': request.form.get('Route3'),
            'Route4': request.form.get('Route4'),
            'Route5': request.form.get('Route5')
        }

        return render_template('home.html', results_indian=result_text, results_us=us_dollar, form_values=form_values)

if __name__ == "__main__":
    application.run(host="0.0.0.0")