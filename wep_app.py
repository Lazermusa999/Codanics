#import libraries
import numpy as np 
import streamlit as st 
import plotly.express as px 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write('''
# Explore Different ML Models
### Dekhty ha konsa ha best ha in ma sa? ''' )

with st.sidebar:
    dataset_name = st.selectbox('Select Dataset',('Iris','Breast Cancer','Wine'))
    classifier_name = st.selectbox('Select Classifier',('KNN','SVM','Random Forest'))
    def get_dataset(dataset_name):
        data=None
        if dataset_name == 'Iris':
            data=datasets.load_iris()
        elif dataset_name == 'Breast Cancer':
            data = datasets.load_breast_cancer()
        else:
            data=datasets.load_wine()
        X = data.data
        y = data.target
        return X,y
X,y=get_dataset(dataset_name) 
st.write('Shape of Dataset:', X.shape)
st.write('Numer of classes:', len(np.unique(y)))   
def get_parameter_ui(classifier_name):
        params  = dict() #get empty dictionary
        if classifier_name == 'KNN':
            K = st.sidebar.slider('K',1,15)
            params['K'] = K #its the number of nearest neighbours

        elif classifier_name == 'SVM':
            C = st.sidebar.slider('C',0.01,10.0,value=1.0)
            params['C'] = C   #its the degree of correct classification.
        else:
            max_depth = st.sidebar.slider('max_depth',2,15)
            params['max_depth'] = max_depth #depth of tree  
            n_estimators = st.sidebar.slider('n_estimators',1,100)
            params['n_estimators'] = n_estimators # the number of trees. 
        return params
params=get_parameter_ui(classifier_name)
def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1)
    return clf
clf = get_classifier(classifier_name, params)

#Now by splitting the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)
clf.fit(X_train,y_train)
y_pred= clf.predict(X_test)

acc=accuracy_score(y_test,y_pred)
st.write('Classifier: ',classifier_name)
st.write('Accuracy: ',acc)

pca = PCA(2)
x_projected = pca.fit_transform(X)
x1 = x_projected[:,0]
x2= x_projected[:,1]
fig = plt.figure()

plt.scatter(x1, x2, c=y , alpha=0.8, cmap='viridis')

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.colorbar()
st.pyplot(fig)