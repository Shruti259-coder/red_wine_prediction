****
**WINE QUALITY PREDICTION**

Import requried library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')
from sklearn.model_selection import KFold,cross_val_score
my_KFold=KFold(n_splits=5)
Loading datasets
wine=pd.read_csv("/content/winequality-red.csv")
print("Successfully import data")
print(wine)
sns.heatmap(wine.isnull())
**DISPLOT**
sns.displot(wine['alcohol'])
**HISTOGRAM**
wine.hist(figsize=(10,10),bins=50)
plt.show()
**PAIR PLOT**
sns.pairplot(wine,hue="quality")
x=wine.iloc[:,:-1].values
y=wine.iloc[:,-1].values
print(y)
**FEATURES IMPORTANCES**
*SPLITTING DATASET*
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)
*USING LOGISTIC REGRESSION*
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy score:",accuracy_score(y_test,y_pred))
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)
*USING KNN ALGORITHM*
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
myknnsc=cross_val_score(model,x_test,y_test,cv=my_KFold)
print(myknnsc)
print("Mean of Kfold"+str(myknnsc.mean()))
*USING GAUSSIAN NB*
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(x_train,y_train)
y_pred3=model3.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy score:",accuracy_score(y_test,y_pred3))
*DECISION TREE ALGORITHM*
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
ypred=dt.predict(x_test)
ac=accuracy_score(y_test,ypred)
print(ac)
*RANDOM FOREST ALGORITHM*
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
ypred=rf.predict(x_test)
ac=accuracy_score(y_test,ypred)
print(ac)
*SVM LINEAR ALGORITHM*
from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)
ypred=svm_model.predict(x_test)
ac=accuracy_score(y_test,ypred)
print(ac)
