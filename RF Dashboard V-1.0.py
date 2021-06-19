import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import subprocess

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

X,y=make_moons(n_samples=300,noise=0.2,random_state=20)
X_train,X_test,y_train,y_test=train_test_split(X,y)


st.sidebar.markdown("# Support Vector Regression Dashboard")

# dataset=st.sidebar.selectbox(
#     'Dataset',
#     ('DS1','DS2')
# )

tuning=st.sidebar.radio(
    'Hyper Parameter',
    ('Default Value','Tuning')
)
if tuning=='Tuning':
    st.sidebar.info('Initially, all parameter values are set to default. Change them according to your need.:smiley:')
    loss=st.sidebar.selectbox(
        'Loss Function',
        ('deviance','exponential')
    )

    learning_rate=st.sidebar.number_input('Learning Rate',value=0.1)

    n_estimators=st.sidebar.slider('Boosting Stages (n_estimators)',min_value=1,max_value=500,value=100)

    subsample=st.sidebar.number_input('Subsample',min_value=0.01,max_value=1.0,value=1.0)

    criterion=st.sidebar.selectbox(
        'Criterion',
        ('friedman_mse','mse','mae')
    )

    min_samples_split=st.sidebar.slider('Minimum Samples Split',value=2,min_value=2)

    min_samples_leaf=st.sidebar.slider('Minimum Samples Leaf',min_value=1,value=1)

    min_weight_fraction_leaf=st.sidebar.number_input('Minimum Weight Fraction',min_value=0.0,max_value=1.0,value=0.0)

    max_depth=st.sidebar.slider('Maximum Depth',min_value=2,value=3)

    min_impurity_decrease=st.sidebar.number_input('Minimum Impurity Decrease',min_value=0.0,max_value=1.0,value=0.0)

    min_impurity_split=st.sidebar.selectbox('Minimum Impurity Split',
                                            ('None','Value Input')
                                            )
    if min_impurity_split=='None':
        min_impurity_split=None
    else:
        min_impurity_split=st.sidebar.number_input('Value',min_value=0.0000000,value=0.0000000,step=0.0000001,format='%.7f')

    init=st.sidebar.selectbox(
        'Init',
        ('zero','estimator')
    )
    if init=='estimator':
        st.sidebar.error('This feature can only be used if you have made another model, whose outcome is to be used as the initial estimates of your Gradient Boosting model.')
        st.info("Set 'Init' parameter value to 'zero'")

    random_state=st.sidebar.selectbox('Random State',
                                      ('None','Value Input')
                                      )

    if random_state=='None':
        random_state=None
    else:
        random_state=st.sidebar.slider('Random State',min_value=1,value=1)

    max_features=st.sidebar.selectbox(
        'Max Features',
        ('None','auto','sqrt','log2'),
    )
    if max_features=='None':
        max_features=None

    verbose=st.sidebar.slider('Verbose (Printing has to be noted)',min_value=0,value=0)

    max_leaf_nodes=st.sidebar.selectbox('Maximum Leaf Nodes',
                                        ('None','Value Input')
                                        )
    if max_leaf_nodes=='None':
        max_leaf_nodes=None
    else:
        max_leaf_nodes=st.sidebar.slider('Value',min_value=1)

    warm_start=st.sidebar.selectbox(
        'Warm Start',
        ('False','True')
    )

    validation_fraction=st.sidebar.number_input('Validation Fraction',min_value=0.0,max_value=1.0,value=0.1)

    n_iter=st.sidebar.selectbox('n Iteration No Change',
                         ('None','Value')
                         )
    if n_iter=='Value':
        n_iter_no_change= st.sidebar.slider('Value')
    else:
        n_iter_no_change=None

    tol=st.sidebar.number_input('Tolerance',min_value=0.0000,value=0.0001,step=0.0001,format='%.4f')

    ccp_alpha=st.sidebar.number_input('Cost-Complexity Pruning Alpha',value=0.0,min_value=0.0000)

    clf = GradientBoostingClassifier(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split,
                                     min_samples_leaf, min_weight_fraction_leaf, max_depth ,min_impurity_decrease,
                                     min_impurity_split, init, random_state, max_features, verbose, max_leaf_nodes,
                                     warm_start, validation_fraction,n_iter_no_change,tol,ccp_alpha)
else:
    clf=SVR()


fig,ax=plt.subplots()

ax.scatter(X.T[0],X.T[1],c=y,cmap='rainbow')
orig=st.pyplot(fig)


if st.sidebar.button('Run Algorithm'):
    with st.spinner('Your model is getting trained..:muscle:'):
        orig.empty()

        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)

        XX,YY,input_array=draw_meshgrid()
        labels=clf.predict(input_array)

        ax.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.3,cmap='rainbow')

        plt.xlabel('Col1')
        plt.ylabel('Col2')
        orig=st.pyplot(fig)
        st.sidebar.subheader("Accuracy of the model: "+str(round(accuracy_score(y_test,y_pred),2)))
    st.success("Done!")
# subprocess = subprocess.Popen(clf,shell=True, stdout=subprocess.PIPE)
# subprocess_return = subprocess.stdout.read()
# print(subprocess_return)