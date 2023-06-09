import pandas as pd
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import class_weight

# Function to evaluate predicted versus actual values
def evauation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val),2)
  score_MAE = round(mean_absolute_error(pred, y_val),2)
  score_r2score = round(r2_score(pred, y_val),2)
  return score_MSE, score_MAE, score_r2score

# Function to fit a model and return a performance measure
def models_score(model_name, train_data, y_train, val_data, y_val):
    model_list = ["Decision_Tree", "Random_Forest", "XGboost_Regressor"]
    # model_1
    if model_name == "Decision_Tree":
        reg = DecisionTreeRegressor(random_state=42)
    # model_2
    elif model_name == "Random_Forest":
        reg = RandomForestRegressor(random_state=42)

    # model_3
    elif model_name == "XGboost_Regressor":
        reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, )
    else:
        print("please enter correct regressor name")

    if model_name in model_list:
        reg.fit(train_data, y_train)
        pred = reg.predict(val_data)

        score_MSE, score_MAE, score_r2score = evauation_model(pred, y_val)
        return round(score_MSE, 2), round(score_MAE, 2), round(score_r2score, 2)

# Load dataframe
data = pd.read_csv("defaults_data.csv")

pd.options.display.max_columns = data.shape[1]

# Recode EDUCATION, SEX, MARRIAGE
education_mapping = {0:'other'
                    ,1:'graduate school'
                    ,2:'university'
                    ,3:'high school'
                    ,4:'other'
                    ,5:'other'
                    ,6:'other'}

data = data.assign(EDUCATION=data.EDUCATION.map(education_mapping))

sex_mapping = {1:'male'
              ,2:'female'}

data = data.assign(SEX=data.SEX.map(sex_mapping))

marriage_mapping = {0:'other'
                   , 1:'married'
                   , 2:'single'
                   , 3:'other'}

data = data.assign(MARRIAGE=data.MARRIAGE.map(marriage_mapping))

# Identify dependent variable
y = data['default']
data_clean = data.drop("default", axis=1)

#%%
# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(data_clean,y, test_size=0.2, random_state=42)

# Encode SEX, EDUCATION, MARRIAGE
label_encoder = LabelEncoder()
x_train['SEX'] = label_encoder.fit_transform(x_train['SEX'].values)
x_test['SEX'] = label_encoder.transform(x_test['SEX'].values)
x_train['EDUCATION'] = label_encoder.fit_transform(x_train['EDUCATION'].values)
x_test['EDUCATION'] = label_encoder.transform(x_test['EDUCATION'].values)
x_train['MARRIAGE'] = label_encoder.fit_transform(x_train['MARRIAGE'].values)
x_test['MARRIAGE'] = label_encoder.transform(x_test['MARRIAGE'].values)

# List of different models to try
model_list = ["Decision_Tree","Random_Forest","XGboost_Regressor"]
#%%
# Fit the different models and get performance scores for each
result_scores = []
for model in model_list:
    score = models_score(model, x_train, y_train, x_test, y_test)
    result_scores.append((model, score[0], score[1],score[2]))
    print(model,score)

df_result_scores = pd.DataFrame(result_scores,columns=["model","mse","mae","r2score"])
df_result_scores
#%%
num_estimator = [50, 150, 200, 500]

# Hyperparamaters to test
space = {'max_depth': hp.quniform("max_depth", 7, 28, 1),
         'gamma': hp.uniform('gamma', 0, 9),
         'reg_alpha': hp.quniform('reg_alpha', 30, 180, 1),
         'reg_lambda': hp.uniform('reg_lambda', 0, 5),
         'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 200),
         'n_estimators': hp.choice("n_estimators", num_estimator),
         }

# Function to test hyperparameters
def hyperparameter_tuning(space):
    model = xgb.XGBClassifier(n_estimators=space['n_estimators'], max_depth=int(space['max_depth']),
                             gamma=space['gamma'],
                             reg_alpha=int(space['reg_alpha']), min_child_weight=space['min_child_weight'],
                             colsample_bytree=space['colsample_bytree'], objective="reg:squarederror",
                              scale_pos_weight=3.5)

    score_cv = cross_val_score(model, x_train, y_train, cv=5, scoring="roc_auc").mean()
    return {'loss': -score_cv, 'status': STATUS_OK, 'model': model}

# Fit models with different hyperparameters
# Identify best model
trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print(best)
#%%
# Create weights
#classes_weights = class_weight.compute_sample_weight(
#    class_weight='balanced',
#    y=y_train
#)

# Code to weight the model to account for imballance
# , sample_weight=classes_weights

# Fit XGBoost model with best hyperparameters
best['max_depth'] = int(best['max_depth']) # convert to int
best["n_estimators"] = num_estimator[best["n_estimators"]] # assing n_estimator because it returs the index
best_xgboost_model = xgb.XGBRegressor(**best)
best_xgboost_model.fit(x_train,y_train)
pred = best_xgboost_model.predict(x_test)
score_MSE, score_MAE, score_r2score = evauation_model(pred,y_test)
to_append = ["XGboost_hyper_tuned",score_MSE, score_MAE, score_r2score]
df_result_scores.loc[len(df_result_scores)] = to_append

# Save the best XGBoost model
best_xgboost_model.save_model("best_model.json")
#%%