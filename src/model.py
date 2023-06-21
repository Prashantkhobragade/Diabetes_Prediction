#importing Library


import pandas as pd
import numpy as np
import pickle
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


#data preprocessing
def preprocess(data:pd.DataFrame)->pd.DataFrame:
    """
    This function will preprocess the dataset.
        
    """
    #reading the dataset
    df = pd.read_csv('C:\Users\sony\Desktop\Diabetes_prediction\Diabetes_Prediction\dataset\diabetes (1).csv')

    #renaming the DiabetesPedigreeFunction column as DPF
    df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

    ## Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
    df['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] = df['Glucose', 'BloodPress', 'SkinThickness', 
                                                                    'Insulin', 'BMI'].replace(0, np.NaN)
    return df


#Creating Dependened and Independent Features

X = df.drop('Outcome', axis=1)  #Independent Features
y = df['Outcome']   #Dependent Features




def get_data_transformer_object(cls)->Pipeline:
    
    robust_scaler = RobustScaler()
    simple_imputer = SimpleImputer(strategy="mean")
    preprocessor = Pipeline(
        steps=[
            ("Imputer", simple_imputer), #replace missing values with zero
            ("RobustScaler", robust_scaler) #keep every feature in same range and handle outlier
            ]
        )
            
    return preprocessor

X_mean = mean_pipeline.fit_transform(X)

def data_transformer():
    #Resampling the minority class
    smt = SMOTETomek(random_state=42, sampling_strategy='minority',n_jobs=-1)
    #fit the model to generate the data
    X_res, y_res = smt.fit_resample(X_mean,y)
    return X_res, y_res

def train_test_split(X_res, y_res):
    X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test


#model 
def train_model(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model


#create Model pickle object
filename = 'xgb_clf.pkl'
pickle.dump(model, open(filename, 'wb'))

