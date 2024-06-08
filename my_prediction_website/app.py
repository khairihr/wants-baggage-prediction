from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np

app = Flask(__name__)

# Load original data and the model
df_original = pd.read_csv('D:\khrd\Semester akhir S1\Data Science\my_prediction_website\customer_booking.csv', encoding='latin1')
rf_model = joblib.load('models/rf_model_wants_extra_baggage.pkl')

# Get feature names used during training (excluding the target)
feature_names_used_in_training = rf_model.feature_names_in_

# Preprocess the original data to match the training data
df_original.drop(['booking_complete', 'flight_duration', 'route', 'wants_extra_baggage'], axis=1, inplace=True)  # Drop irrelevant columns and the target
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_cols = encoder.fit_transform(df_original[['booking_origin']])
encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(['booking_origin']))
df_original = pd.concat([df_original.select_dtypes(exclude='object'), encoded_df], axis=1)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input
        sales_channel = request.form['sales_channel']
        trip_type = request.form['trip_type']
        flight_day = request.form['flight_day']
        route = request.form['route']
        booking_origin = request.form['booking_origin']

        # Create DataFrame from user input with *all* features
        input_data = pd.DataFrame({
            'num_passengers': [int(request.form['num_passengers'])],
            'sales_channel': [sales_channel],
            'trip_type': [trip_type],
            'purchase_lead': [int(request.form['purchase_lead'])],
            'length_of_stay': [int(request.form['length_of_stay'])],
            'flight_hour': [int(request.form['flight_hour'])],
            'flight_day': [flight_day],
            'route': [route],
            'booking_origin': [booking_origin]
        })

        # Map 'Yes' and 'No' to 1 and 0 for 'wants_preferred_seat' and 'wants_in_flight_meals'
        input_data['wants_preferred_seat'] = 1 if request.form['wants_preferred_seat'] == 'Yes' else 0
        input_data['wants_in_flight_meals'] = 1 if request.form['wants_in_flight_meals'] == 'Yes' else 0


        # One-hot encode categorical features
        encoded_input = encoder.transform(input_data[['booking_origin']]).toarray()
        encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['booking_origin']))
        input_data = pd.concat([input_data.drop('booking_origin', axis=1), encoded_df], axis=1)

        # Reorder columns to match training data
        input_data = input_data[feature_names_used_in_training]

        # Make prediction
        prediction = rf_model.predict_proba(input_data)[0][1] 

        # Determine if wants extra baggage or not
        wants_baggage = "wants extra baggage" if prediction > 0.5 else "doesn't want extra baggage"

        return render_template('result.html', prediction=prediction, wants_baggage=wants_baggage, model='Random Forest')  

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
