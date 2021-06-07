import pandas as ps 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.metrics import precision_recall_curve, classification_report

#Loading the data





df = ps.read_csv('C:/Users/Administrator.LAPTOP-C7V2HJKO/Desktop/учебка/6_сем/machine_learning/2lab/pulsar_data_train.csv')

df = df.dropna()
df = df.drop(' Excess kurtosis of the integrated profile', 1)
df = df.drop(' Excess kurtosis of the DM-SNR curve', 1) 
df = df.drop(' Skewness of the integrated profile', 1)
df = df.drop(' Skewness of the DM-SNR curve', 1)

dfT = df[df.target_class == 1.0]
dfT = dfT.head(500)
dfF = df[df.target_class == 0.0]
dfF = dfF.head(500)

df = ps.concat([dfT, dfF])
df = df.sample(frac = 1)

target = df['target_class']
data = df.drop('target_class', 1)

# data = data.to_numpy()
# target = target.to_numpy()

# target = np.array(list(map(lambda x: 1 if np.bool(x) else 0, target)))

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)
rf = RandomForestClassifier(max_depth=2)
rf = rf.fit(X_train, y_train)
print('Random forest:')
print(classification_report(y_test, rf.predict(X_test), target_names=['Non-pulsar', 'Pulsar']))

dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
print('Decision tree:')
print(classification_report(y_test, dtree.predict(X_test), target_names=['Non-pulsar', 'Pulsar']))

lg = LogisticRegression(random_state=0).fit(X_train, y_train)
print('Logictic regression:')
print(classification_report(y_test, lg.predict(X_test), target_names=['Non-pulsar', 'Pulsar']))
