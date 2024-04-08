import streamlit as st

st.title('การทำนายการลาออกของพนักงาน')
st.header('นาย ชินวัฒน์ ภูไชยแสง')
st.subheader('สาขาวิชาวิทยาการข้อมูล')
st.markdown("----")

col1, col2 = st.columns(2)
#col1.write("This is column 1")
#col2.write("This is column 2")
with col1:
    st.image('./pic/Resignation.png')
with col2:
    st.image('./pic/Resignation2.png')

html_1 = """
<div style="background-color:#52BE80;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ข้อมูลของพนักงาน</h5></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

import pandas as pd

raw_data=pd.read_csv('./data/Employee.csv')
st.write(raw_data.head(10))

html_2 = """
<div style="background-color:#52BE80;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>แปลงข้อมูลให้พร้อมสำหรับการทำนาย</h5></center>
</div>
"""
st.markdown(html_2, unsafe_allow_html=True)
st.markdown("")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
raw_data["Education"]= le.fit_transform(raw_data["Education"])
raw_data["City"]=le.fit_transform(raw_data["City"])
raw_data["Gender"]=le.fit_transform(raw_data["Gender"])
raw_data["EverBenched"]=le.fit_transform(raw_data["EverBenched"])

st.write(raw_data.head(10))

html_3 = """
<div style="background-color:#FFBF00;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายคลาสการลาออกของพนักงาน</h5></center>
</div>
"""
st.markdown(html_3, unsafe_allow_html=True)
st.markdown("")   


s1 = st.number_input("กรุณาเลือกข้อมูล Education", step=1, format="%d")
s2 = st.slider("กรุณาเลือกข้อมูล JoiningYear",2012,2018)
s3 = st.number_input("กรุณาเลือกข้อมูล City", step=1, format="%d")
s4 = st.number_input("กรุณาเลือกข้อมูล PaymentTier", step=1, format="%d")
s5 = st.number_input("กรุณาเลือกข้อมูล Age", step=1, format="%d")
s6 = st.number_input("กรุณาเลือกข้อมูล Gender", step=1, format="%d")
s7 = st.number_input("กรุณาเลือกข้อมูล EverBenched", step=1, format="%d")
s8 = st.number_input("กรุณาเลือกข้อมูล ExperienceInCurrentDomain", step=1, format="%d")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

if st.button("ทำนายผล"):
    X= raw_data.drop(columns='LeaveOrNot')
    y=raw_data['LeaveOrNot']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.3)

    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)

    #ข้อมูล input สำหรับทดลองจำแนกข้อมูล
    x_input = np.array([[s1, s2, s3, s4, s5, s6, s7, s8]])
    st.write(rf_model.predict(x_input))
    out=rf_model.predict(x_input)

    if out[0]== 0:
     st.header("อยู่ต่อ")
    elif out[0]== 1:
      st.header("มีแนวโน้มว่าจะลาออก")
      st.button("ไม่ทำนายผล")
else :
 st.button("ไม่ทำนายผล")