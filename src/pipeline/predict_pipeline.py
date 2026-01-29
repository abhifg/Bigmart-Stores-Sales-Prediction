import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=(model.predict(data_scaled))**8
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 item_type: str,
                 item_fat_content:str,
                 item_weight:float,
                 item_visibility:float,
                 item_mrp:float,
                 outlet_size:str,
                 outlet_type:str,
                 outlet_location_type:str,
                 outlet_age:int):
        self.item_type=item_type
        self.item_fat_content=item_fat_content
        self.item_weight=item_weight
        self.item_visibility=item_visibility
        self.item_mrp=item_mrp
        self.outlet_size=outlet_size
        self.outlet_type=outlet_type
        self.outlet_location_type=outlet_location_type
        self.oulet_age=outlet_age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "Item_Type":[self.item_type],
                "Item_Fat_Content":[self.item_fat_content],
                "Item_Weight":[self.item_weight],
                "Item_Visibility":[self.item_visibility],
                "Item_MRP":[self.item_mrp],
                "Outlet_Size":[self.outlet_size],
                "Outlet_Type":[self.outlet_type],
                "Outlet_Location_Type":[self.outlet_location_type],
                "Outlet_Age":[self.oulet_age]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        

                 