import pandas as pd
import numpy as np
from Model import Model
from sklearn.metrics import recall_score, f1_score
from ExtractSupportVector import supportVec
from sklearn import preprocessing
df=pd.read_csv("DataSet.csv",",")


le=preprocessing.LabelEncoder()
for x in df.columns:
   if df[x].dtype==object:
       df[x]=le.fit_transform(df[x])


sizeAll,e=df.shape
trainSize=int(0.2*sizeAll)
Train=df[df.index<=trainSize]

Test=df[df.index>trainSize]
y=Train["isFraud"]
X=Train.drop("isFraud",1)
y_test=Test["isFraud"]
X_test=Test.drop("isFraud",1)

X1,X2=supportVec(X,y,Train,"isFraud")
print("TrainP_size",Train["isFraud"].sum())
print("XPS.shape",X1.shape)
print("X2",X2.shape)
'''
NewTrain=
model=Model(X,y,10)
_, accuracy = model.evaluate(X_test, y_test)
prediction=model.predict(X_test)
prediction=np.round(prediction)
print("prediction",f1_score(y_test,prediction))

print("accuracy",accuracy)
print(max(prediction))'''
