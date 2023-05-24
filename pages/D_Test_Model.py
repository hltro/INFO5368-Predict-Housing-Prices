import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt        # pip install matplotlib
import streamlit as st                  # pip install streamlit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pages.A_Explore_Dataset import convert_df
from pages.C_Train_Model import split_dataset
import random
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 2 - Predicting Housing Prices Using Regression")

#############################################

st.title('Test Model')

#############################################
def restore_dataset():
    """
    Input: 
    Output: 
    """
    df=None
    if 'house_df' not in st.session_state:
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
        else:
            return None
    else:
        df = st.session_state['house_df']
    return df

def mae(y_true, y_pred):
    """
    Measures the absolute difference between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - mean absolute error
    """
    mae_score=-1

    # Add code here
    mae_score = mean_absolute_error(y_true, y_pred)
    
    # st.write('rmse not implemented yet.')
    return mae_score

def rmse(y_true, y_pred):
    """
    This function computes the root mean squared error. 
    Measures the difference between predicted and 
    actual values using Euclidean distance.

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - root mean squared error
    """
    rmse_score=-1

    # Add code here
    rmse_score = mean_squared_error(y_true, y_pred, squared=False)
    
    # st.write('rmse not implemented yet.')
    return rmse_score

def r2(y_true, y_pred):
    """
    Compute Coefficient of determination (R2 score). 
    Rrepresents proportion of variance in predicted values 
    that can be explained by the input features.

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - r2 score
    """
    r2=0
    
    # Add code here
    r2 = r2_score(y_true, y_pred)
    
    # st.write('r2 not implemented yet.')
    return r2

# Used to access model performance in dictionaries
METRICS_MAP = {
    'mean_absolute_error': mae,
    'root_mean_squared_error': rmse,
    'r2_score': r2
}

# Checkpoint 9
def compute_eval_metrics(X, y_true, model, metrics):
    """
    This function checks the metrics of the models

    Input:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evlauate performance 
    Output:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {}

    # Add code here
    train_rmse_errors, train_mae_errors, train_r2_errors = 0,0,0

    model.fit(X, y_true)
    y_train_predict = model.predict(X)

    train_rmse_errors = rmse(y_true, y_train_predict)
    train_mae_errors = mae(y_true, y_train_predict)
    train_r2_errors = r2(y_true, y_train_predict)

    if 'root_mean_squared_error' in metrics:
        metric_dict = { 
            'mean_absolute_error':train_mae_errors,'root_mean_squared_error': train_rmse_errors,'r2_score':train_r2_errors
    }

    # st.write('compute_eval_metrics not implemented yet.')
    return metric_dict

def plot_learning_curve(X_train, X_val, y_train, y_val, trained_model, metrics, model_name):
    """
    This function plots the learning curve. Note that the learning curve is calculated using 
    increasing sizes of the training samples
    Input:
        - X_train: training features
        - X_val: validation/test features
        - y_train: training targets
        - y_val: validation/test targets
        - trained_model: the trained model to be calculated learning curve on
        - metrics: a list of metrics to be computed
        - model_name: the name of the model being checked
    Output:
        - fig: the plotted figure
        - df: a dataframe containing the train and validation errors, with the following keys:
            - df[metric_fn.__name__ + " Training Set"] = train_errors
            - df[metric_fn.__name__ + " Validation Set"] = val_errors
    """
    fig = make_subplots(rows=len(metrics), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
    df = pd.DataFrame()

    # Add code here
    train_rmse_errors, train_mae_errors, train_r2_errors = [], [], []
    val_rmse_errors, val_mae_errors, val_r2_errors = [], [], []
    stepSize = []

    for m in range(500, len(X_train)+1, 500):
        trained_model.fit(X_train[:m], y_train[:m])
        y_train_predict = trained_model.predict(X_train[:m])
        #print(y_train_predict.shape)
        y_val_predict = trained_model.predict(X_val)
        stepSize.append(m)

        if 'mean_absolute_error' in metrics:
            train_mae = mae(y_train[:m], y_train_predict)
            val_mae = mae(y_val, y_val_predict)

            train_mae_errors.append(train_mae)
            val_mae_errors.append(val_mae)

        if 'root_mean_squared_error' in metrics:
            train_rmse = rmse(y_train[:m], y_train_predict)
            val_rmse = rmse(y_val, y_val_predict)

            train_rmse_errors.append(train_rmse)
            val_rmse_errors.append(val_rmse)

        if 'r2_score' in metrics:
            train_r2 = r2(y_train[:m], y_train_predict)
            val_r2 = r2(y_val, y_val_predict)

            train_r2_errors.append(train_r2)
            val_r2_errors.append(val_r2)

    df['mae' + " Training Set"] = train_mae_errors
    df['mae' + " Validation Set"] = val_mae_errors
    df['rmse' + " Training Set"] = train_rmse_errors
    df['rmse' + " Validation Set"] = val_rmse_errors
    df['r2' + " Training Set"] = train_r2_errors
    df['r2' + " Validation Set"] = val_r2_errors

    fig.add_trace(go.Scatter(x=stepSize, y=train_mae_errors, name='Train'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stepSize, y=val_mae_errors, name='Validation'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stepSize, y=train_rmse_errors, name='Train'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stepSize, y=val_rmse_errors, name='Validation'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stepSize, y=train_r2_errors, name='Train'), row=3, col=1)
    fig.add_trace(go.Scatter(x=stepSize, y=val_r2_errors, name='Validation'),row=3, col=1)
    
    fig.update_layout(
    title={
        'text': '{}'.format(model_name)
    })
    
    # st.write('plot_learning_curve not implemented yet.')
    return fig, df

# Helper function
def restore_data(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Input: 
        - df: the pandas dataframe
    Output: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    # Restore train/test dataset
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if (X_train is None):
        # Select variable to explore
        numeric_columns = list(df.select_dtypes(include='number').columns)
        feature_select = st.selectbox(
            label='Select variable to predict',
            options=numeric_columns,
        )
        X = df.loc[:, ~df.columns.isin([feature_select])]
        Y = df.loc[:, df.columns.isin([feature_select])]

        # Split train/test
        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

        X_train, X_val, y_train, y_val = split_dataset(X, Y, number)
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

# Helper function
@st.cache
def convert_df(df):
    """
    Cache the conversion to prevent computation on every rerun

    Input: 
        - df: pandas dataframe
    Output: 
        - Save file to local file system
    """
    return df.to_csv().encode('utf-8')

###################### FETCH DATASET #######################
df = None
# df = restore_dataset()
# df = ... Add code here: read in data and store in st.session_state
# df = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
if 'data' in st.session_state:
    df = st.session_state['data']
# st.session_state['df'] = df

if df is not None:
    # Restore dataset splits
    X_train, X_val, y_train, y_val = restore_data(df)

    st.markdown("## Get Performance Metrics")
    metric_options = ['mean_absolute_error',
                      'root_mean_squared_error', 'r2_score']
    
    # Select multiple metrics for evaluation
    metric_select = st.multiselect(
        label='Select metrics for regression model evaluation',
        options=metric_options,
    )
    if (metric_select):
        st.session_state['metric_select'] = metric_select
        st.write('You selected the following metrics: {}'.format(metric_select))

    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']
    trained_models = [
        model for model in regression_methods_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    # Select a trained regression model for evaluation
    model_select = st.multiselect(
        label='Select trained regression models for evaluation',
        options=trained_models
    )
    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected regression models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('## Review Regression Model Performance')

            review_model_select = st.multiselect(
                label='Select models to review',
                options=model_select
            )

            plot_options = ['Learning Curve', 'Metric Results']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=plot_options
            )

            if 'Learning Curve' in review_plot:
                for model in review_model_select:
                    trained_model = st.session_state[model]
                    fig, df = plot_learning_curve(
                        X_train, X_val, y_train, y_val, trained_model, metric_select, model)
                    st.plotly_chart(fig)

            if 'Metric Results' in review_plot:
                models = [st.session_state[model]
                          for model in review_model_select]

                train_result_dict = {}
                val_result_dict = {}
                for idx, model in enumerate(models):
                    train_result_dict[review_model_select[idx]] = compute_eval_metrics(
                        X_train, y_train, model, metric_select)
                    val_result_dict[review_model_select[idx]] = compute_eval_metrics(
                        X_val, y_val, model, metric_select)

                st.markdown('### Predictions on the training dataset')
                st.dataframe(train_result_dict)

                st.markdown('### Predictions on the validation dataset')
                st.dataframe(val_result_dict)

    ###################### DEPLOY MODEL #######################

    # Select a model to deploy from the trained models
    st.markdown("## Choose your Deployment Model")
    model_select = st.selectbox(
        label='Select the model you want to deploy',
        options=st.session_state['trained_models'],
    )

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        st.session_state['deploy_model'] = st.session_state[model_select]

    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(df)
    
    st.session_state['data'] = df

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Test: Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',        
    )

    st.write('Continue to Deploy Model')