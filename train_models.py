import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

def train_heart_disease():
    print("--- Training Heart Disease Model ---")
    num_features = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    cat_features = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 
                    'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 
                    'Asthma', 'KidneyDisease', 'SkinCancer']
    
    # MOCK DATA GENERATOR
    data = {
        'HeartDisease': np.random.choice(['No', 'Yes'], 100),
        'BMI': np.random.uniform(15, 40, 100),
        'PhysicalHealth': np.random.randint(0, 30, 100),
        'MentalHealth': np.random.randint(0, 30, 100),
        'SleepTime': np.random.uniform(4, 12, 100),
        'AgeCategory': np.random.choice(['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'], 100),
        'Race': np.random.choice(['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'], 100),
        'GenHealth': np.random.choice(['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], 100)
    }
    for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']:
        data[col] = np.random.choice(['Yes', 'No'] if col != 'Sex' else ['Male', 'Female'], 100)
        
    df = pd.DataFrame(data)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X, y)
    
    with open('heart_model.pkl', 'wb') as f: pickle.dump(model, f)
    print("Saved: heart_model.pkl")

def train_diabetes():
    print("--- Training Diabetes Model ---")
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = pd.DataFrame(np.random.rand(100, 8), columns=cols)
    y = np.random.randint(0, 2, 100)
    model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression())])
    model.fit(X, y)
    with open('diabetes_model.pkl', 'wb') as f: pickle.dump(model, f)
    print("Saved: diabetes_model.pkl")

def train_breast_cancer():
    print("--- Training Breast Cancer Model ---")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression())])
    model.fit(X, y)
    with open('breast_cancer_model.pkl', 'wb') as f: pickle.dump(model, f)
    print("Saved: breast_cancer_model.pkl")

if __name__ == '__main__':
    train_heart_disease()
    train_diabetes()
    train_breast_cancer()
