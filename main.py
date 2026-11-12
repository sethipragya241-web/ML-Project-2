import pandas as pd
df=pd.read_csv("Titanic-Dataset (1).csv")
print(df.head())
print(df['Survived'].value_counts())
print(df.isna().sum())
print(df.shape)
print(df['Pclass'].value_counts())
df['Gender']=df['Gender'].map({"male":1,"female":0})
df['Embarked']=df['Embarked'].fillna(df['Embarked'].ffill())
print(df['Embarked'].value_counts())
df['Embarked']=df['Embarked'].map({"S":0,"C":1,"Q":2})
x=df[['Pclass','Gender','SibSp','Parch','Embarked']]
y=df['Survived']
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(y_train.shape)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
acc=accuracy_score(y_test,y_pred)
print(acc)
cm=confusion_matrix(y_test,y_pred)
print(cm)
