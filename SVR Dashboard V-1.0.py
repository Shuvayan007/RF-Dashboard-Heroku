import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


X,y=load_diabetes(return_X_y=True)
df=pd.DataFrame(X)
df['target']=y
X=df[9].values.reshape(442,1)
y=df['target'].values
X_train,X_test,y_train,y_test=train_test_split(X,y)


st.sidebar.markdown("# Support Vector Regression Dashboard")

tuning=st.sidebar.radio(
    'Hyper Parameter',
    ('Default Value','Tuning')
)
if tuning=='Tuning':
    st.sidebar.info('Initially, all parameter values are set to default. Change them according to your need.:smiley:')
    kernel=st.sidebar.selectbox(
        'Kernel Type (Precomputed!!)',
        ('rbf','linear','poly','sigmoid','precomputed')
    )

    if kernel=='poly':
        degree=st.sidebar.slider('Degree of the polynomial',value=3, max_value=10)
    else:
        degree=0

    if kernel=='rbf' or kernel=='poly' or kernel=='sigmoid':
        gamma = st.sidebar.selectbox(
            'Gamma',
            ('scale', 'auto', 'float')
        )

        if gamma=='float':
            gamma=st.sidebar.number_input('Gamma Value',value=0.01)
    else:
        gamma='scale'

    if kernel=='poly' or kernel=='sigmoid':
        coef0=st.sidebar.number_input('Coefficient 0')
    else:
        coef0=0.0

    tol=st.sidebar.number_input('Tolerance',value=0.001,format="%.3f",min_value=0.001,step=0.001)

    C=st.sidebar.number_input('C (Regularization Parameter)',min_value=0.01,value=1.0)

    epsilon=st.sidebar.number_input('Epsilon Value',min_value=0.00 ,value=0.1)

    shrinking=st.sidebar.selectbox(
        'Shrinking',
        ('True','False')
    )

    if shrinking=='True':
        shrinking=True
    else:
        shrinking=False

    st.sidebar.beta_expander(':bulb: Tips',expanded=False,
                             ':bulb: If the number of iterations is large, then shrinking can shorten the training time. However, if we loosely solve the optimization problem (e.g., by using a large stopping tolerance), the code without using shrinking may be much faster.')

    st.sidebar.info(':bulb: If the number of iterations is large, then shrinking can shorten the training time. However, if we loosely solve the optimization problem (e.g., by using a large stopping tolerance), the code without using shrinking may be much faster.')

    cache_size=st.sidebar.number_input('Cache Size (In MB)',value=200,min_value=1,step=1)

    verbose=st.sidebar.selectbox(
        'Verbose (Catch output)',
        ('False','True')
    )
    if verbose=='True':
        verbose=True
    else:
        verbose=False

    max_iter=st.sidebar.slider('Maximum Iteration (Left)',value=-1,max_value=500,min_value=-1)

    clf=SVR(kernel,degree,gamma,coef0,tol,C,epsilon,shrinking,cache_size,verbose,max_iter)
else:
    clf=SVR()


fig,ax=plt.subplots()

ax.scatter(X,y)
orig=st.pyplot(fig)


if st.sidebar.button('Run Algorithm'):
    with st.spinner('Your model is getting trained..:muscle:'):
        orig.empty()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)


        x=np.linspace(X.min(),X.max(),X.shape[0]).reshape(X.shape)
        ax.plot(x,clf.predict(x),color='red')

        plt.xlabel('Col1')
        plt.ylabel('Col2')
        orig=st.pyplot(fig)
        st.sidebar.subheader("Mean absolute error of the model: "+str(round(mean_absolute_error(y_test,y_pred),2)))
    st.success("Done!")