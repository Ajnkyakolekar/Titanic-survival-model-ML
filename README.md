# Titanic-survival-model-ML
import pandas as pd
data=pd.read_csv("train.csv")
data=data.dropna(axis=0)
y=data.Survived
features=['Pclass','Age','Fare',]
x= data[features]
from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y=train_test_split(x,y,test_size=0.5)
from sklearn.ensemble import RandomForestClassifier
data_model= RandomForestClassifier(max_depth=10, n_estimators=1000,random_state=1)
data_model.fit(train_x,train_y)
val_ypred= data_model.predict(val_x)
from sklearn.metrics import accuracy_score
B=accuracy_score(val_y, val_ypred)
print(B)

data2=pd.read_csv('test.csv')
X=data2.[features]
Y=data_model.predict(X)
