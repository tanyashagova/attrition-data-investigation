import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.model_selection import KFold


# parameters
C = 0.29
penalty = 'l2'
solver = 'liblinear'
n_splits = 5
output_file = 'model.bin'


# Data preparation

df = pd.read_csv('HR Employee Attrition.csv')
# work with categoracal features
dict_education = {1: 'below_college', 2: 'college', 3: 'bachelor',
                  4: 'master', 5: 'doctor'}
# the same for JobInvolvement, JobSatisfaction, RelationshipSatisfaction
dict_envsat = {1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}
dict_performrat = {1: 'low', 2: 'good', 3: 'excellent', 4: 'outstanding'}
dict_wlbalance = {1: 'bad', 2: 'good', 3: 'better', 4: 'best'}
dict_stockOptionLevel = {0: '0l', 1: '1l', 2: '2l', 3: '3l'}
dict_JobLevel = {1: '1l', 2: '2l', 3: '3l', 4: '4l', 5: '5l'}

df.Education = df.Education.map(dict_education)
df.EnvironmentSatisfaction = df.EnvironmentSatisfaction.map(dict_envsat)
df.JobInvolvement = df.JobInvolvement.map(dict_envsat)
df.JobSatisfaction = df.JobSatisfaction.map(dict_envsat)
df.RelationshipSatisfaction = df.RelationshipSatisfaction.map(dict_envsat)
df.PerformanceRating = df.PerformanceRating.map(dict_performrat)
df.WorkLifeBalance = df.WorkLifeBalance.map(dict_wlbalance)
df.StockOptionLevel = df.StockOptionLevel.map(dict_stockOptionLevel)
df.JobLevel = df.JobLevel.map(dict_JobLevel)

target = ['Attrition']
categorical = list(df.dtypes[df.dtypes == 'object'].index.values)
categorical.remove('Attrition')
categorical.remove('Over18')

numerical = list(df.dtypes[df.dtypes == 'int64'].index.values)
numerical.remove('EmployeeCount')
numerical.remove('StandardHours')


for col in categorical + target:
    df[col] = df[col].str.lower().str.replace('&','').str.replace('  ',' ').str.replace(' ','_')
    
df.Attrition = (df.Attrition == 'yes').astype(int)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_full_train.reset_index(drop=True)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.Attrition
y_val = df_val.Attrition
y_test = df_test.Attrition

del df_train['Attrition']
del df_val['Attrition']
del df_test['Attrition']


# train model
initial_model = LogisticRegression(solver=solver, C=C, max_iter=1000, penalty=penalty)

def train_model(model, df_train, y_train):
    
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model.fit(X_train, y_train)
    
    return dv, model 

def predict(dv, model, df_val):
    
    dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    y_pred = model.predict_proba(X_val)[:, 1]
    
    return y_pred


# validation 
print('validation')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []


for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.Attrition
    y_val = df_val.Attrition
    
    dv, model = train_model(initial_model, df_train, y_train)
    y_pred = predict(dv, model, df_val)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    

print(f'auc score = {np.mean(scores)} +- {np.std(scores)}')


# training final model and validate it on test data
print('train final model')
dv, model = train_model(initial_model, df_full_train, df_full_train.Attrition)
y_pred = predict(dv, model, df_test)
auc = roc_auc_score(y_test, y_pred)
print('auc on test data: ', auc)


# saving final model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print(f'model is saved in {output_file}')

