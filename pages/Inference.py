import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import pickle

def app():
    st.title("Inference")
    csv = st.file_uploader("Upload a csv file of well data", type="csv")
    if csv is not None:
        df = pd.read_csv(csv)
        
        # Preprocess the data
        drop_columns = ['Unnamed: 0', 'pad_id', 'standardized_operator_name', 'average_stage_length', 'average_proppant_per_stage', 'average_frac_fluid_per_stage', 'number_of_stages', 'frac_type']
        df.drop(columns=drop_columns, inplace=True)
        df.replace([np.inf, -np.inf, 'Undefined', 'Unknown'], np.nan, inplace=True)

        categorical_columns = ['ffs_frac_type', 'relative_well_position', 'batch_frac_classification', 'well_family_relationship']

        # Impute nan values
        imputer = SimpleImputer(strategy='constant', fill_value='missing')
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

        # Ordinal encode categorical columns
        encoder = pickle.load(open('Model/encoder.pkl', 'rb'))
        df[categorical_columns] = encoder.transform(df[categorical_columns])

        # Calculate the 'well_skew' feature
        # The formula for well skew is sqrt(|surface_x - bh_x|^2 + |surface_y - bh_y|^2)
        df['well_skew'] = np.sqrt(np.square(df['surface_x'] - df['bh_x']) + np.square(df['surface_y'] - df['bh_y']))

        # Create a 'lateral_length' feature
        # The formula for lateral length is sqrt(|horizontal_toe_x - bh_x|^2 + |horizontal_toe_y - bh_y|^2)
        df['lateral_length'] = np.sqrt(np.square(df['horizontal_toe_x'] - df['bh_x']) + np.square(df['horizontal_toe_y'] - df['bh_y']))

        # Load the KNN model
        KNN_model = pickle.load(open('Model/KNN_final.pkl', 'rb'))

        # Get the KNN predictions
        df['KNN_OilPeakRate'] = KNN_model.predict(df[['surface_x', 'surface_y']])

        # Load the AdaBoost model
        final_model = pickle.load(open('Model/final_model.pkl', 'rb'))

        # Get the final predictions
        y_pred = final_model.predict(df)

        # Create a dataframe of the predictions
        df_pred = pd.DataFrame(y_pred, columns=['OilPeakRate'])

        # Give option to download the predictions
        pred_csv = df_pred.to_csv(index=False)
        st.download_button(label='Download Predictions', data=pred_csv, file_name='predictions.csv', mime='text/csv')

        # Display the csv file
        st.dataframe(df_pred)

if __name__ == "__main__":
    app()