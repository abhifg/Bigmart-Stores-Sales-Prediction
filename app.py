from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline




app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        

        data = CustomData(
            item_type=request.form.get("item_type"),  # matches 'Dairy', 'Breads', etc.
            item_fat_content=request.form.get("item_fat_content"),  # 'Regular', 'Low Fat', 'LF'
            item_weight=float(request.form.get("item_weight")),
            item_visibility=(float(request.form['item_visibility'])/100)**(1/3),
            item_mrp=float(request.form.get("item_mrp")),
            outlet_size=request.form.get("outlet_size"),
            outlet_type=request.form.get("outlet_type"),
            outlet_location_type=request.form.get("outlet_location_type"),
            outlet_age=2026 - int(request.form.get("outlet_est_year"))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        output=round(results[0],2)
        return render_template('form.html',prediction_text="Item-Outlet Sales will be around: ₹{} per day".format(output))
    else:
        return render_template('form.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5002,debug=True)



