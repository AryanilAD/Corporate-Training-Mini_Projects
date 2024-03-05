from pywebio.input import input, NUMBER
from pywebio.output import put_text
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Load the Boston housing dataset
boston = fetch_openml(data_id=531)
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('house_price_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Define the predict function
def predict_price():
    input_data = {
        'CRIM': input('CRIM (per capita crime rate by town): ', type=NUMBER),
        'ZN': input('ZN (proportion of residential land zoned for lots over 25,000 sq.ft.): ', type=NUMBER),
        'INDUS': input('INDUS (proportion of non-retail business acres per town): ', type=NUMBER),
        'CHAS': input('CHAS (Charles River dummy variable): ', type=NUMBER),
        'NOX': input('NOX (nitric oxides concentration): ', type=NUMBER),
        'RM': input('RM (average number of rooms per dwelling): ', type=NUMBER),
        'AGE': input('AGE (proportion of owner-occupied units built prior to 1940): ', type=NUMBER),
        'DIS': input('DIS (weighted distances to five Boston employment centres): ', type=NUMBER),
        'RAD': input('RAD (index of accessibility to radial highways): ', type=NUMBER),
        'TAX': input('TAX (full-value property-tax rate per $10,000): ', type=NUMBER),
        'PTRATIO': input('PTRATIO (pupil-teacher ratio by town): ', type=NUMBER),
        'B': input('B (1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town): ', type=NUMBER),
        'LSTAT': input('LSTAT (% lower status of the population): ', type=NUMBER)
    }

    put_text("Input data:")
    for key, value in input_data.items():
        put_text(f"{key}: {value}")

    input_features = np.array([input_data[col] for col in boston.feature_names]).reshape(1, -1)
    put_text("Input features:")
    put_text(input_features)

    predicted_price = model.predict(input_features)[0]

    put_text(f"Predicted house price: ${predicted_price:.2f}")

# Run the application
if __name__ == "__main__":
    predict_price()
