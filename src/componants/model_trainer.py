import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Linear Regression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "Random Forest":RandomForestRegressor(),
                "XGBOOST":XGBRegressor()
            }
            params={
                "Linear Regression":{},
                "Lasso":{"alpha":[0.0001,0.0005,0.0009,0.01,0.05,0.09,0.1,0.5,0.9]},
                "Ridge":{"alpha":[0.0001,0.0005,0.0009,0.01,0.05,0.09,0.1,0.5,0.9]},
                "Random Forest":{
                    'n_estimators':[100,150,200,250],
                    'max_depth': [5,6,7,8],
                    'min_samples_split': [8,9,10,11],
                    'min_samples_leaf': [20,22,24,25]
                },
                "XGBOOST":{
                    'n_estimators':[12,13,15,16,17,18,19,20,22,24,25],
                    'max_depth':[4,5,6,7],
                    'booster':['gbtree'],
                    'min_child_weight':[60,65,72,75,76,78,80],
                    'base_score':[3,4,5,6,7,8],
                    'lambda':[2,4,6,8,10,12],
                    'alpha':[3,6,8,9,12,14]
                }
            }
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models,params)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            ##if best_model_score<0.6:
                ##raise CustomException("No best model found",sys)
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_score_=r2_score(y_test,predicted)
            return r2_score_,best_model
        
        except Exception as e:
            raise CustomException(e,sys)
