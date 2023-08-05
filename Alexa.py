# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:36:36 2023

@author: nayan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 6 00:22:29 2022

@author: hp

"""
#Implementing Logistic Regression


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Alexa_dataset.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#splitting into trainig and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

#Classifier -Logistic Regression
#fitting Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#predicting the test results
y_pred=classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

#Performance
#Performance metrics
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_lr = accuracy_score(y_test, y_pred)
print('Accuracy is %f' % accuracy_lr)

#Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is %f' % precision)
 
# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is %f' % recall)

# Function definition for visualization of results
def Visualizer(argument1, arguement2):
  from matplotlib.colors import ListedColormap
  X_set, y_set = argument1, arguement2
  X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop        
  =X_set[:,0].max()+1, step=0.01), 
  np.arange(start=X_set[:, 1].min() - 1, stop = 
  X_set[:, 1].max() + 1, step = 0.01))
  plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), 
  X2.ravel()]).T).reshape(X1.shape),
  alpha= 0.75, cmap = ListedColormap(('red', 'green')))

  plt.xlim(X1.min(), X1.max())
  plt.ylim(X2.min(), X2.max())
  for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label=j)
  plt.xlabel('Age')
  plt.ylabel('Minutes_of_Music_Consumed')
  plt.legend()
  plt.show()

"""1.9: Visualizing the Training set results"""

#Visualizing the Training set results
Visualizer(X_train, y_train)

"""1.10: Visualizing the Test set results"""

#Visualizing the Test set results
Visualizer(X_test, y_test)


#2.) Implementing K-nearest neighbours

#import libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv('Alexa_dataset.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#training and testing 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting k-NN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

#predicting the model
y_pred=classifier.predict(X_test)

#making the confusion metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)

#Performance metrics
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_knn = accuracy_score(y_test, y_pred)
print('Accuracy is %f' % accuracy_knn)

#Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is %f' % precision)
 
# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is %f' % recall)

Visualizer(X_train, y_train)
Visualizer(X_test, y_test)

# Classifier – III  (Naïve Bayes Classifier)
# Fitting Naïve Bayes to the Training set 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

"""3.6: Predicting the Test set results"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""3.7: Making the confusion matrix

"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""3.8: Performance metrics"""

#Performance metrics
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_nbc = accuracy_score(y_test, y_pred)
print('Accuracy is %f' % accuracy_nbc)
	 
# Precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precision is %f' % precision)
 
# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Recall is %f'% recall)

Visualizer(X_train, y_train)
Visualizer(X_test, y_test)

"""3.9: Visualizing the Training set results

"""

#Visualizing the Training set results

"""3.10: Visualizing the Test set results"""

#Visualizing the Test set results


scores = [accuracy_lr,accuracy_knn,accuracy_nbc]
algorithms = ["Logistic Regression","K-Nearest Neighbors","Naive-Bayes Classifier"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i]))
    
import seaborn as sns
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)