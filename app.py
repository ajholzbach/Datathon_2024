import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

#
# Data Loading
#

# Load the data
training_data = pd.read_csv('Data/training.csv')
training_data.drop(columns=['Unnamed: 0'], inplace=True)
training_data.drop_duplicates(inplace=True)
training_data.replace([np.inf, -np.inf], np.nan, inplace=True)
# Convert any "Undefined" or "Unknown" values to NaN
training_data.replace('Undefined', np.nan, inplace=True)
training_data.replace('Unknown', np.nan, inplace=True)
# Before imputing, keep track of rows with missing values for 'OilPeakRate' to remove them later
rows_with_missing_values = training_data[training_data['OilPeakRate'].isnull()].index.to_list()
training_data = training_data.drop(rows_with_missing_values)
location_features = ['surface', 'bh', 'horizontal_midpoint', 'horizontal_toe']
x_cols = [lf + '_x' for lf in location_features]
y_cols = [lf + '_y' for lf in location_features]
x_min = training_data[x_cols].min().min()
x_max = training_data[x_cols].max().max()
y_min = training_data[y_cols].min().min()
y_max = training_data[y_cols].max().max()
knn_values = [3, 5, 10, 15]

#
# KNN mesh creation
#

X = training_data[['surface_x', 'surface_y']]
y = training_data['OilPeakRate'] 
knn_results = []
for num in knn_values:
    new_model = KNeighborsRegressor(num = n)
    new_model.fit(X, y)
    xx = np.linspace(x_min, x_max, 1000)
    yy = np.linspace(x_min, x_max, 1000)
    xx, yy = np.meshgrid(xx, yy)
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    pred = new_model.predict(X_grid)
    knn_results.append(pred)

# Predict the values of the mesh
y_pred = model.predict(X_grid)
#
# Site Stuff
#

st.markdown("<h1 style=\"font-family: 'Helvetica'; font-size: 3em; text-align: center;\">\
Well Tempered</h1>"
    ,unsafe_allow_html=True)
st.markdown("<h1 style=\"font-family: 'Helvetica'\">Well Locations</h1>"
    ,unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    feature_used = st.radio('Location:', location_features, index=0)
with col2:
    knn_dist = st.radio('KNN:', [None, *knn_values], index=0)

fig = px.scatter(training_data, x=feature_used + '_x', y=feature_used + '_y',
    color='OilPeakRate',
    color_continuous_scale=px.colors.sequential.Agsunset,
    hover_data={'pad_id': True,
                'OilPeakRate': False,
                feature_used + '_x': False,
                feature_used + '_y': False}
)
fig.update_xaxes(range=[x_min, x_max])
fig.update_yaxes(range=[y_min, y_max])
fig.update_layout(
    plot_bgcolor='#0f1116',  # Dark plot background
    paper_bgcolor='#0f1116',  # Dark around the plot
    font_color="white",  # White font for contrast
    yaxis=dict(
        showgrid=False
    )
)
# Optionally, if you have additional information to show on hover:
fig.update_traces(marker=dict(size=3),
    hoverinfo='text',
    hovertext=[f'Additional info for point {i}' for i in range(len(training_data))]
)
st.plotly_chart(fig, use_container_width=True)