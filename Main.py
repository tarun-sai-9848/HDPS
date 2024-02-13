import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
def rf():
    from sklearn.ensemble import RandomForestClassifier
    model= RandomForestClassifier()
    model.fit(X_train,Y_train)
    X_train_prediction = model.predict(X_train)
    tda_rf = accuracy_score(X_train_prediction, Y_train)
    y.append(tda_rf)
    input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),
    int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0]==1:
        pred_list.append(1)
    else:
        pred_list.append(0)
def dt():
    from sklearn.tree import DecisionTreeClassifier
    model= DecisionTreeClassifier()
    model.fit(X_train,Y_train)
    X_train_prediction = model.predict(X_train)
    tda_dt = accuracy_score(X_train_prediction, Y_train)
    y.append(tda_dt)
    input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),
    int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0]==1:
        pred_list.append(1)
    else:
        pred_list.append(0)


def svm():
    from sklearn.svm import SVC
    model = SVC(C=1,kernel='linear',gamma="auto")
    model.fit(X_train, Y_train)
    X_train_prediction = model.predict(X_train)
    tda_svm = accuracy_score(X_train_prediction, Y_train)
    y.append(tda_svm)
    input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),int(restecg),
    int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0]==1:
        pred_list.append(1)
    else:
        pred_list.append(0)


def nn():
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(100,50),
                        max_iter = 500,activation = 'relu',
                        solver = 'adam')
    model.fit(X_train, Y_train)
    X_train_prediction = model.predict(X_train)
    tda_nn = accuracy_score(X_train_prediction, Y_train)
    y.append(tda_nn)
    input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),
    int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0]==1:
        pred_list.append(1)
    else:
        pred_list.append(0)


def nb():
    from sklearn.naive_bayes import GaussianNB
    model=GaussianNB()
    model.fit(X_train, Y_train)
    X_train_prediction = model.predict(X_train)
    tda_nb = accuracy_score(X_train_prediction, Y_train)
    y.append(tda_nb)
    input_data = (int(age),int(gender),int(cp),int(bp),int(chol),int(fbs),
    int(restecg),int(thalch),int(exang),int(oldpeak),int(slope),int(ca),int(thal))
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0]==1:
        pred_list.append(1)
    else:
        pred_list.append(0)

        
warnings.filterwarnings('ignore')
heart_data = pd.read_csv("heart.csv")
X = heart_data.drop(columns='target', axis=1)
df=pd.DataFrame(heart_data)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
st.title("Heart disease prediction system")
form = st.sidebar.form(key='my_form')
age=form.number_input(label="Age")
gender=form.radio("Gender",["Male","Female"])
cp=form.number_input(label="Chest Pain")
bp=form.number_input(label = "Enter Rest blood pressure")
chol=form.number_input(label = "Enter Serum cholesterol")
fbs=form.number_input(label = "Enter Fasting blood sugar")
restecg=form.number_input(label = "Enter Rest electrocardiograph")
thalch=form.number_input(label = "Enter MaxHeart rate")
exang=form.number_input(label = "Enter Exercise-induced angina")
oldpeak=form.number_input(label = "Enter ST depression")
slope=form.number_input(label = "Enter slope")
ca=form.number_input(label = "Enter No. of vessels ")
thal=form.number_input(label = "Enter thalassemia")
submit_button = form.form_submit_button(label='Submit')
if gender=="Male":
    gender=1
else:
    gender=0
if submit_button:
    pred_list=[]
    y=[]
    rf()
    dt()
    svm()
    nn()
    nb()
    if pred_list.count(1)>pred_list.count(0):
        st.subheader("Result: Positive")
    else:
        st.subheader("Reslut: Negative")
    from bokeh.plotting import figure
    x = ['Random Forest','Decision Tree' ,'SVM' ,'NN' ,'Naive Bayes' ]
    p = figure(x_range=x, height=350, title="Accuracy",toolbar_location=None, tools="")
    p.vbar(x=x, top=y, width=0.5)
    st.bokeh_chart(p, use_container_width=True)