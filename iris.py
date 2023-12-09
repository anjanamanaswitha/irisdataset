
actual file:
https://colab.research.google.com/drive/1jQP8faTq0pAiVM4Z4JdAbcCNG34VDY_q?usp=sharing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Iris.csv')
df

df.shape

df.columns=[1,2,3,4,5,6]
df.columns

plt.figure(figsize=(11, 6))
sns.lineplot(x=5, y=2, data=df)
plt.title("line Plot of Sepal Length by Species")
plt.show()

plt.figure(figsize=(11, 6))
sns.violinplot(x=4, y=3, data=df)
plt.title("Violin Plot of Petal Width by Species")
plt.show()

plt.figure(figsize=(6, 6))
correl_matrix =df.corr()
sns.heatmap(correl_matrix, annot=True, cmap='gist_heat', linewidths=.7)
plt.title("Correlation Heatmap of Iris Dataset Features")
plt.show()

plt.figure(figsize=(8, 6))
species_counts = df[6].value_counts()
species_counts.plot(kind='barh', color=['blue', 'green', 'yellow'])
plt.title("Samples by Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

x=df.iloc[:,1:5]
y=df.iloc[:,5]
print(x)
print('..............................................')
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
print("x_train")
print(x_train)
print('-------------------------------',x_train.shape)
print("x_test")
print(x_test)
print('-------------------------------',x_test.shape)
print("y_train")
print(y_train)
print('-------------------------------',y_train.shape)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)


print(y_test)
print('......................................')
y_pred


from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
accuracy=accuracy_score(y_test,y_pred)*100
accuracy

x_new=np.array([[2.4,5.6,7.9,3.0]])
y_new=knn.predict(x_new)
y_new

x_new=np.array([[1.6,2.6,1.5,2.5]])
print(x_new)
y_new=knn.predict(x_new)
y_new

