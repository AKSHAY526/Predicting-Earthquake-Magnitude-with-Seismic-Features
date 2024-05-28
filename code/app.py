from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
app = Flask(__name__, template_folder='templates')


# Load the dataset
dataset = pd.read_csv("query.csv")

# Remove null values
dataset.dropna(inplace=True)
# Remove null values
dataset.dropna(inplace=True)

# Convert 'time' column to datetime format
dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')


# Select features and target variable
X = dataset[['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError',
             'magNst']]
Y = dataset['mag']

# Split the dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor()

# Fit the model
model.fit(X_train, Y_train)


# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')
@app.route('/datasource.html')
def datasource():
    return render_template('datasource.html')
@app.route('/graphs.html')
def show_graphs():
    # Generate and render graphs
    histogram_depth_img = generate_histogram('depth', 'Depth Histogram', 'Depth (km)', 'Frequency')
    histogram_magnitude_img = generate_histogram('mag', 'Magnitude Histogram', 'Magnitude', 'Frequency')
    scatter_lat_lon_img = generate_scatter_plot('latitude', 'longitude', 'Latitude vs. Longitude', 'Latitude', 'Longitude')
    scatter_depth_mag_img = generate_scatter_plot('depth', 'mag', 'Depth vs. Magnitude', 'Depth (km)', 'Magnitude')
    time_series_mag_img = generate_time_series_plot('time', 'mag', 'Magnitude Time Series', 'Time', 'Magnitude')
    earthquakes_per_year_img = generate_earthquakes_per_year_plot()
    scatter_lat_lon_depth_mag_img = generate_scatter_plot_lat_lon_depth_mag()

    return render_template('graphs.html',
                           histogram_depth_img=histogram_depth_img,
                           histogram_magnitude_img=histogram_magnitude_img,
                           scatter_lat_lon_img=scatter_lat_lon_img,
                           scatter_depth_mag_img=scatter_depth_mag_img,
                           time_series_mag_img=time_series_mag_img,
                           earthquakes_per_year_img=earthquakes_per_year_img,
                           scatter_lat_lon_depth_mag_img=scatter_lat_lon_depth_mag_img)

def generate_earthquakes_per_year_plot():
    plt.figure(figsize=(10, 6))
    dataset['time'] = pd.to_datetime(dataset['time'])
    dataset['year'] = dataset['time'].dt.year
    earthquakes_per_year = dataset['year'].value_counts().sort_index()
    earthquakes_per_year = earthquakes_per_year.reindex(range(1960, 2025), fill_value=0)
    plt.bar(earthquakes_per_year.index, earthquakes_per_year.values, color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Number of Earthquakes')
    plt.title('Number of Earthquakes per Year')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'


def generate_scatter_plot_lat_lon_depth_mag():
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset['longitude'], dataset['latitude'], s=dataset['depth'], c=dataset['mag'], cmap='viridis', alpha=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Latitude vs Longitude with Depth and Magnitude')
    plt.colorbar(label='Magnitude')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'



# Function to generate histogram and return it as a base64 encoded image
def generate_histogram(column, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.hist(dataset[column], bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'


# Function to generate scatter plot and return it as a base64 encoded image
def generate_scatter_plot(x_column, y_column, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset[x_column], dataset[y_column])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'


# Function to generate time series plot and return it as a base64 encoded image
def generate_time_series_plot(x_column, y_column, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.plot(dataset[x_column], dataset[y_column])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'


@app.route('/predict.html')
def predict_html():
    return render_template('predict.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        depth = float(request.form['depth'])
        nst = float(request.form['nst'])
        gap = float(request.form['gap'])
        dmin = float(request.form['dmin'])
        rms = float(request.form['rms'])
        horizontalError = float(request.form['horizontalError'])
        depthError = float(request.form['depthError'])
        magError = float(request.form['magError'])
        magNst = float(request.form['magNst'])

        # Predict using the trained model
        prediction = model.predict(
            [[latitude, longitude, depth, nst, gap, dmin, rms, horizontalError, depthError, magError, magNst]])

        # Process prediction for display
        if prediction[0] < 1.0:
            category = "Micro"
            effects = "Generally not felt by people, though recorded on local instruments. More than 100,000 earthquakes per year."
        elif 1.0 <= prediction[0] < 3.0:
            category = "Micro"
            effects = "Generally not felt by people, though recorded on local instruments. More than 100,000 earthquakes per year."
        elif 3.0 <= prediction[0] < 4.0:
            category = "Minor"
            effects = "Felt by many people; no damage. (12,000 to 100,000) earthquakes per year."
        elif 4.0 <= prediction[0] < 5.0:
            category = "Light"
            effects = "Felt by all; minor breakage of objects. (2,000 to 12,000) earthquakes per year."
        elif 5.0 <= prediction[0] < 6.0:
            category = "Moderate"
            effects = "Some damage to weak structures. (200 to 2,000) earthquakes per year."
        elif 6.0 <= prediction[0] < 7.0:
            category = "Strong"
            effects = "Moderate damage in populated areas. (20 to 200) earthquakes per year."
        elif 7.0 <= prediction[0] < 8.0:
            category = "Major"
            effects = "Serious damage over large areas; loss of life. (3 to 20) earthquakes per year."
        elif prediction[0] >= 8.0:
            category = "Great"
            effects = "Severe destruction and loss of life over large areas. Fewer than 3 earthquakes per year."
        else:
            category = "Unknown"
            effects = "Category not defined."

        value = round(prediction[0], 2)
        return render_template('predict.html', prediction=f'Predicted Magnitude: {value} ({category})',
                               effects=f'Effects: {effects}')


if __name__ == '__main__':
    app.run(debug=True)
