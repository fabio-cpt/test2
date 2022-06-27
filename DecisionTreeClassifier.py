# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:51:35 2022

@author: Fabio Caputo
"""


from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasRegressor # this should be teh wrapper to run tensorflow with sklearn pipelines

from sklearn.tree import DecisionTreeClassifier

df = data_1.copy()

df.columns

clf_model = DecisionTreeClassifier(
    criterion="gini",  # "entropy" / "gini"
    random_state=42,
    max_depth=3,
    min_samples_leaf=5)   


df = df[['transactions_past_month', 'transactions_past_3_month','churn_flag']]

df.info()

df.isnull().any()

clf_model.fit(df.loc[:,[x for x in df if x != "churn_flag"]],df["churn_flag"].astype(float))

y_predict = clf_model.predict(df.loc[:,[x for x in df if x != "churn_flag"]])

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(df["churn_flag"].astype(float),y_predict)

# plotting decision tree

def stringify(x):
    return str(x)

target = list(df["churn_flag"].unique())
result = map(stringify,target)
target = [x for x in result]

feature_names = list(df.loc[:,[x for x in df if x != "churn_flag"]].columns)

from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(clf_model
                      ,out_file=None
                      ,feature_names=feature_names  
                      ,class_names=target
                      # ,filled=True
                      # ,rounded=True
                      # ,special_characters=True
                      )  

graph = graphviz.Source(dot_data)  
graph


from sklearn.tree import export_text
r = export_text(clf_model, feature_names=feature_names)
print(r)

# graph.save('graph1.jpg')




