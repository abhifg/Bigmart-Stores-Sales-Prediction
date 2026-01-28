import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts","preprocessor.pkl")

class ItemWeightImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.means_=X.groupby("Item_Type")["Item_Weight"].mean()
        return self
    
    def transform(self,X):
        X=X.copy()
        for item,mean in self.means_.items():
            mask=(X["Item_Type"]==item)&(X["Item_Weight"].isnull())
            X.loc[mask,"Item_Weight"]=mean
        return X

class OutletSizeImputer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.outlet_size_map = {
            "Supermarket Type1": "Small",
            "Supermarket Type2": "Medium",
            "Supermarket Type3": "Medium",
            "Grocery Store": "Small"
        }
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X=X.copy()
        for outlet,size in self.outlet_size_map.items():
            mask=(X["Outlet_Type"]==outlet)&(X["Outlet_Size"].isnull())
            X.loc[mask,"Outlet_Size"]=size
        return X
class ItemFatContentCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["Item_Fat_Content"] = X["Item_Fat_Content"].replace({
            "LF": "Low Fat",
            "low fat": "Low Fat",
            "Regular": "Regular",
            "reg": "Regular"
        })
        return X


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")
            numerical_columns=['Item_Weight','Item_Visibility','Item_MRP','Outlet_Age']
            categorical_columns=["Item_Fat_Content","Item_Type","Outlet_Location_Type","Outlet_Size","Outlet_Type"]

            pre_pipeline = Pipeline(
                steps=[
                    ("item_weight_imputer", ItemWeightImputer()),
                    ("outlet_size_imputer", OutletSizeImputer()),
                    ("fat_cleaner", ItemFatContentCleaner())
                    
                ]
            )

            num_pipeline=Pipeline(
                steps=[
                    
                    ("scaler",StandardScaler())
                ]
            )

            cat_onehot_columns=["Item_Fat_Content","Item_Type","Outlet_Type"]
            cat_ordinal_columns=["Outlet_Location_Type","Outlet_Size"]

            ordinal_pipeline=Pipeline(
                [
                    
                    ("ordinalencoder",OrdinalEncoder(categories=[['Tier 1','Tier 2','Tier 3'],['Small','Medium']],handle_unknown="use_encoded_value",unknown_value=-1))
                ]
            )

            onehot_pipeline=Pipeline(
                [
                    
                    ("onehotencoder",OneHotEncoder())
                ]
            )


            logging.info("Numerical Columns Standard Scaling completed")
            logging.info("Categorical Columns Encoding completed")


            preprocessor=Pipeline(
                steps=[
                    ("pre_pipeline",pre_pipeline),
                    ("column_transformer",
                
                
                ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("onehot_pipeline",onehot_pipeline,cat_onehot_columns),
                    ("ordinal_pipeline",ordinal_pipeline,cat_ordinal_columns)
                ]
            ))]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_coloumn_name="Item_Outlet_Sales"
            numerical_columns=['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']

            input_feature_train_df=train_df.drop(columns=[target_coloumn_name],axis=1)
            target_feature_train_df=train_df[target_coloumn_name]

            input_feature_test_df=test_df.drop(columns=[target_coloumn_name],axis=1)
            target_feature_test_df=test_df[target_coloumn_name]

            logging.info("Applying preprocessing object on training and testing datasets")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
