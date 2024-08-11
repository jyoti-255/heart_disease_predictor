#import lib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#load the data
data=pd.read_csv("hd_march24.csv")
print(data)

#check for null data
print(data.isnull().sum())

#check for duplicate data
print(data.duplicated().sum())

#features and target
'''jabhi jyada  features rahnge to rather than writting all the features it's better to drop the output ka column and phir other'''

features=data.drop("output",axis="columns")
target=data["output"]
print(features)
print(target)
#train and test
x_train,x_test,y_train,y_test=train_test_split(features.values,target)

#model
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

#performances
cr=classification_report(y_test,model.predict(x_test))
print(cr)

#prediction
d=[[  63 ,1 ,  3  ,   145,   233,    1 ,0  ,150  , 0  ,2.3  ,0   ,0  ,1   ]]
res=model.predict(d)
print(res)




