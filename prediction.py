import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evauation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val),2)
  score_MAE = round(mean_absolute_error(pred, y_val),2)
  score_r2score = round(r2_score(pred, y_val),2)
  return score_MSE, score_MAE, score_r2score

data = pd.read_csv("defaults_data.csv")

y = data['default']

data_clean = data.drop("default", axis=1)

x_train, x_test, y_train, y_test = train_test_split(data_clean, y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
x_train['SEX'] = label_encoder.fit_transform(x_train['SEX'].values)
x_test['SEX'] = label_encoder.transform(x_test['SEX'].values)
x_train['EDUCATION'] = label_encoder.fit_transform(x_train['EDUCATION'].values)
x_test['EDUCATION'] = label_encoder.transform(x_test['EDUCATION'].values)
x_train['MARRIAGE'] = label_encoder.fit_transform(x_train['MARRIAGE'].values)
x_test['MARRIAGE'] = label_encoder.transform(x_test['MARRIAGE'].values)

#save label encoder classes
np.save('classes.npy', label_encoder.classes_)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")
pred = best_xgboost_model.predict(x_test)
score_MSE, score_MAE, score_r2score = evauation_model(pred, y_test)
print(score_MSE, score_MAE, score_r2score)
#%%
loaded_encoder = LabelEncoder()
loaded_encoder.classes_ = np.load('classes.npy',allow_pickle=True)
print(x_test.shape)
input_default = loaded_encoder.transform(np.expand_dims("0",-1))
print(int(input_default))
inputs = np.expand_dims([0,1,1,1,24,0,0],0)
print(inputs.shape)
prediction = best_xgboost_model.predict(inputs)
print("final pred", np.squeeze(prediction,-1))

