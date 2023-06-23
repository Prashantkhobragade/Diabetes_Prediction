#importing Library
import sys
  


from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
import pickle
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib



#data preprocessing

df = pd.read_csv('C:\\Users\\sony\\Desktop\\Diabetes_prediction\\Diabetes_Prediction\\dataset\\diabetes.csv')

#renaming the DiabetesPedigreeFunction column as DPF
df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

## Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 
                                                                        'Insulin', 'BMI']].replace(0, np.NaN)
    



#Creating Dependened and Independent Features

X = df.drop('Outcome', axis=1)  #Independent Features
y = df['Outcome']   #Dependent Features





    
try:
    robust_scaler = RobustScaler()
    simple_imputer = SimpleImputer(strategy="mean")
    preprocessor = Pipeline(
        steps=[
            ("Imputer", simple_imputer), #replace missing values with zero
            ("RobustScaler", robust_scaler) #keep every feature in same range and handle outlier
            ]
        )
        
    X_mean = preprocessor.fit_transform(X)
                
       

except Exception as e:
    raise CustomException(e,sys)


try:
    #Resampling the minority class
    smt = SMOTETomek(random_state=42, sampling_strategy='minority',n_jobs=-1)
            
    #fit the model to generate the data
    X_res, y_res = smt.fit_resample(X_mean,y)

except Exception as e:
    raise CustomException(e,sys)
        
        


#spliting dataset
X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.2, random_state=42)
        




try:
    #model 
    model = XGBClassifier()
    model.fit(X_train, y_train)

except Exception as e:
    raise CustomException(e,sys)
    

#create Robust Scaler pickle object

filename = 'robust_scaler.pkl'
joblib.dump(robust_scaler,open(filename,'wb'))


#create Model pickle object
filename = 'xgb_clf.pkl'
joblib.dump(model, open(filename, 'wb'))


