import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
dataset = pd.read_csv('D:\miniproject\heart.csv')
dataset2 = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Define features and target variable
cols = ['cp_0', 'cp_1', 'cp_2', 'cp_3', 'trestbps', 'chol', 'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2', 'thalach', 'exang_0', 'exang_1']
X = dataset2[cols]
y = dataset2.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train the model
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)

# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('sc.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Your form processing code here

        # Example: Extract values from the form
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])

        # Example: Create an input array for prediction
        input_features = np.array([[cp == i for i in range(4)] + [trestbps, chol, fbs == 0, fbs == 1, restecg == 0, restecg == 1, restecg == 2, thalach, exang == 0, exang == 1]])

        # Standardize the input features
        input_features = sc.transform(input_features)

        # Make a prediction
        prediction = classifier.predict(input_features)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
