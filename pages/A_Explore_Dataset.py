import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
import tarfile
import urllib.request
from itertools import combinations

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

feature_lookup = {
    'longitude': '**longitude** - longitudinal coordinate',
    'latitude': '**latitude** - latitudinal coordinate',
    'housing_median_age': '**housing_median_age** - median age of district',
    'total_rooms': '**total_rooms** - total number of rooms per district',
    'total_bedrooms': '**total_bedrooms** - total number of bedrooms per district',
    'population': '**population** - total population of district',
    'households': '**households** - total number of households per district',
    'median_income': '**median_income** - median income',
    'ocean_proximity': '**ocean_proximity** - distance from the ocean',
    'median_house_value': '**median_house_value**',
    'city':'city location of house',
    'road':'road of the house',
    'county': 'county of house',
    'postcode':'zip code',
    'rooms_per_household':'average number of rooms per household',
    "number_bedrooms":'number of bedrooms',
    "number_bathrooms": 'number of bathrooms',
    

}
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 2 - Predicting Housing Prices Using Regression")

#############################################

st.markdown('# Explore Dataset')

#############################################

st.markdown('### Import Dataset')
# Task 1: Import dataset
def load_dataset(data):
    """
    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    - Checkpoint 1 - Read .csv file containing a dataset
    """
    # Erro checking - check for string
    df = pd.read_csv(data)
    return df

# def user_input_features(df, chart_type, x, y):
#     """
#     Input: pnadas dataframe containing dataset
#     Output: dictionary of sidebar filters on features
#     """
#     numeric_columns = list(df.select_dtypes(['float','int']).columns)
#     side_bar_data = {}
#     for feature in numeric_columns:
#         try:
#             f = st.sidebar.slider(str(feature), float(df[str(feature)].min()), float(df[str(feature)].max()), float(df[str(feature)].mean()))
#         except Exception as e:
#             print(e)
#         side_bar_data[feature] = f
#     return side_bar_data

# Task 3: Checkpoint 1
def compute_correlation(X, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings 
        with description of magnitude and direction of correlation

    Input: 
        - X: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output: 
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = None
    cor_summary_statements = []

    # Add code here
    correlation = X[features].corr()
    cor = 0
    feature_pairs = combinations(features, 2)
    
    for f1, f2 in feature_pairs:
        cor = correlation[f1][f2]
    if (cor > 0.5):
        # print("- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}")
        cor_summary_statements.append("- Features {} and {} are {} {} correlated: {}".format(f1, f2, "strongly", "positively", round(cor,2)))
    elif (cor < -0.5):
        cor_summary_statements.append("- Features {} and {} are {} {} correlated: {}".format(f1, f2, "strongly", "negatively", round(cor,2)))   
    elif(0 < cor < 0.5):
        cor_summary_statements.append("- Features {} and {} are {} {} correlated: {}".format(f1, f2, "weakly", "positively", round(cor,2)))      
    else:
        cor_summary_statements.append("- Features {} and {} are {} {} correlated: {}".format(f1, f2, "weakly", "negatively", round(cor,2)))
    
    # st.write('compute_correlation not implemented yet.')
    return correlation, cor_summary_statements

# Task 2: Show visualization of features
# Helper Function
def user_input_features(df, chart_type, x, y):
    """
    This function renders the feature selection sidebar 

    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
        - x: features
        - y: targets
    Output: 
        - dictionary of sidebar filters on features
    """
    side_bar_data = []

    select_columns = []
    if (x is not None):
        select_columns.append(x)
    if (y is not None):
        select_columns.append(y)
    if (x is None and y is None):
        select_columns = list(df.select_dtypes(include='number').columns)

    for idx, feature in enumerate(select_columns):
        try:
            f = st.sidebar.slider(
                str(feature),
                float(df[str(feature)].min()),
                float(df[str(feature)].max()),
                (float(df[str(feature)].min()), float(df[str(feature)].max())),
                key=chart_type+str(idx)
            )
        except Exception as e:
            print(e)
        side_bar_data.append(f)
    return side_bar_data

# Helper Function
def display_features(df, feature_lookup):
    """
    This function displayes feature names and descriptions (from feature_lookup).

    Inputs:
        - df (pandas.DataFrame): The input DataFrame to be whose features to be displayed.
        - feature_lookup (dict): A dictionary containing the descriptions for the features.
    Outputs: None
    """
    numeric_columns = list(df.select_dtypes(include='number').columns)
    #for idx, col in enumerate(df.columns):
    for idx, col in enumerate(numeric_columns):
        if col in feature_lookup:
            st.markdown('Feature %d - %s' % (idx, feature_lookup[col]))
        else:
            st.markdown('Feature %d - %s' % (idx, col))

# Helper Function
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    This function fetches a dataset from a URL, saves it in .tgz format, and extracts it to a specified directory path.

    Inputs:
    - housing_url (str): The URL of the dataset to be fetched.
    - housing_path (str): The path to the directory where the extracted dataset should be saved.

    Outputs: None
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

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
# df = ... Add code here: read in data and store in st.session_state

# Create two columns for dataset upload
# Call functions to upload data or restore dataset
col1, col2 = st.columns(2)

with(col1): # uploading from local machine
    data = st.file_uploader('Upload your data', type=['csv','txt'])
with(col2): # upload from cloud
    data_path = st.text_input("Enter data url", "", key="data_url")

    if(data_path):
        fetch_housing_data()
        data = os.path.join(HOUSING_PATH, "housing.csv")
        st.write("You entered: ", data_path)

st.session_state['data'] = None
st.session_state['data'] = data

if data:

    ###################### EXPLORE DATASET #######################
    st.markdown('### Explore Dataset Features')

    # Load dataset
    df = load_dataset(data)
    st.write(df)

    # Restore dataset if already in memory
    st.session_state['data'] = df

    # Display feature names and descriptions (from feature_lookup)
    display_features(df, feature_lookup)

    # Display dataframe as table
    st.dataframe(df.describe())

    ###################### VISUALIZE DATASET #######################
    st.markdown('### Visualize Features')
    
    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection using user_input_features()
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label = 'Types of chart',
        options = [ 'Scatterplot', 'Lineplot', 'Histogram', 'Boxplot']
    )
    st.write(chart_select)
    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    
    # Draw plots including Scatterplot, Histogram, Lineplot, Boxplot
    if chart_select == 'Scatterplot' or chart_select == 'Histogram' or chart_select == 'Lineplot' or chart_select == 'Boxplot':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = user_input_features(df, chart_select, x=x_values, y=y_values)

            if chart_select == 'Scatterplot':
                plot = px.scatter(data_frame=df,
                                  x=x_values,
                                  y=y_values,
                                  range_x=[side_bar_data[0][0], side_bar_data[0][1]],
                                  range_y=[side_bar_data[1][0], side_bar_data[1][1]])

            if chart_select == 'Histogram':
                plot = px.histogram(data_frame=df,
                                  x=x_values,
                                  y=y_values,
                                  range_x=[side_bar_data[0][0], side_bar_data[0][1]],
                                  range_y=[side_bar_data[1][0], side_bar_data[1][1]])

            if chart_select == 'Lineplot':
                plot = px.line(data_frame=df,
                                  x=x_values,
                                  y=y_values,
                                  range_x=[side_bar_data[0][0], side_bar_data[0][1]],
                                  range_y=[side_bar_data[1][0], side_bar_data[1][1]])

            if chart_select == 'Boxplot':
                plot = px.box(data_frame=df,
                                  x=x_values,
                                  y=y_values,
                                  range_x=[side_bar_data[0][0], side_bar_data[0][1]],
                                  range_y=[side_bar_data[1][0], side_bar_data[1][1]])

            st.write(plot)
        except Exception as e:
            print(e)
            
    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### Looking for Correlations")
    # Collect features for correlation analysis using multiselect
    
    # Compute correlation between selected features
    select_features = st.multiselect(
            'Select two or features for correlation',
            options=numeric_columns
    )
    correlation = compute_correlation(df,select_features)
    st.write(correlation) 
    
    # Display correlation of all feature pairs with description of magnitude and direction of correlation
    if(select_features):
        try:
          fig = scatter_matrix(df[select_features], figsize=(12,8)) 
          st.pyplot(fig[0][0].get_figure()) 
        except Exception as e:
            print(e)
            
    # Store dataset in st.session_state
    
    # st.markdown('Continue to Preprocess Data')
    
    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(df)

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',        
    )

    st.markdown('#### Continue to Preprocess Data')