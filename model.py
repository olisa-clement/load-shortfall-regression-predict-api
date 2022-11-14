"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    def preprocess(input_df):
        
        # replace missing values
        mean_val_pres = input_df['Valencia_pressure'].mean()
        input_df['Valencia_pressure'] = input_df['Valencia_pressure'].fillna(mean_val_pres)
        
        # create new features Year, Month, Day, Hour
        df_sub_time = input_df['time'].str.split('[-:\s]', expand=True)
        df_sub_time.rename(columns={0: 'Year', 1: 'Month', 2: 'Day', 3: 'Hour', 4: 'x', 5: 'y'}, inplace=True)
        df_sub_time.drop(['x', 'y'], axis=1, inplace=True)
        input_df = pd.concat([input_df, df_sub_time], axis=1)
        
        # engineer existing features from Valancia_wind_deg and Seville_pressure
        input_df['Valencia_wind_deg'] = input_df['Valencia_wind_deg'].str.extract('(\d+)')
        input_df['Seville_pressure'] = input_df['Seville_pressure'].str.extract('(\d+)')
        
        # Change the object dtypes to numeric
        input_df[['Valencia_wind_deg', 'Seville_pressure', 'Year', 'Month', 'Day', 'Hour']] = input_df[['Valencia_wind_deg', 
                                                                                                            'Seville_pressure', 
                                                                                                            'Year', 'Month', 'Day', 
                                                                                                            'Hour']].apply(pd.to_numeric)
        
        # Drop irrelevant columns to our model
        input_df = input_df.drop([col for col in input_df.columns if col.endswith(('temp_max', 'temp_min'))], axis=1)
        df_clean = input_df.drop(['Unnamed: 0', 'time', ], axis = 1)
        
        return df_clean

    predict_vector = preprocess(feature_vector_df)
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
