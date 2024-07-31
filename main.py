from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Load the dataset to get unique values for dropdowns
data = pd.read_csv('final_dataset.csv')
bedrooms = sorted(data['beds'].unique())
bathrooms = sorted(data['baths'].unique())
sizes = sorted(data['size'].unique())
zip_codes = sorted(data['zip_code'].unique())

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    beds = int(request.form['beds'])
    baths = float(request.form['baths'])
    size = float(request.form['size'])
    zip_code = int(request.form['zip_code'])

    # Prepare the input for the model
    input_data = pd.DataFrame([[beds, baths, size, zip_code]], columns=['beds', 'baths', 'size', 'zip_code'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Convert the price to positive if it's negative
    positive_price = abs(prediction)

    # Return the positive prediction
    return f"Predicted Price: INR {positive_price:.2f}"

if __name__ == '__main__':
    app.run(debug=True)

