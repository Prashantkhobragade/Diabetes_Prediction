# Diabetes_Prediction

## About Project
The data set that has been used in this project has been taken from the Kaggle. This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

### Features of the dataset:
The dataset contains **768** individuals data with **9** features set. The detailed description of all the features is as follows:

**Pregnancies:** indicates the number of pregnancies.

**Glucose:** indicates the plasma glucose concentration

**Blood Pressure:** indicates diastolic blood pressure in mm/Hg

**Skin Thickness:** indicates triceps skinfold thickness in mm

**Insulin:** indicates insulin in U/mL

**BMI:** indicates the body mass index in kg/m2

**Diabetes Pedigree Function:** indicates the function which scores likelihood of diabetes based on family history

**Age:** indicates the age of the person

**Outcome:** indicates if the patient had a diabetes or not (1 = yes, 0 = no)

## DISCLAIMER

This is a POC(Proof of concept) kind-of project. The data used here comes up with no guarantee from the creator. So, don't use it for making any decisions. If you do so, the creator is not responsible for anything. However, this project presents the idea that how we can use ML/DL in Medical Science if developed at large scale and with authentic and verified data


### How to run ?

*  Before the following steps make sure you have `git`, `Anaconda` or `miniconda` installed on your system
*  Clone the complete project with git clone [https://github.com/PrashantKhobragade/Diabetes_Prediction.git](https://github.com/Prashantkhobragade/Diabetes_Prediction) or you can just download the code and unzip it
*  Once the project is cloned, open `anaconda` prompt in the directory where the project was cloned and paste the following block

    `conda create -p venv python=3.9 -y`

    `conda activate venv/`
   
    `pip install -r requirements.txt`

*  And finally run the project with

         `python -m src.model.py`

### Deployment

This website is deployed at [render](render.com)

You can access it [here](https://diabetes-prediction-wvme.onrender.com/)

**Note:** The website may take a minute to load sometimes, as the server may be in hibernate state
