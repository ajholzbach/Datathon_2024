import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from plotly.subplots import make_subplots

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
knn_results = {}
xx = np.linspace(x_min, x_max, 1000)
yy = np.linspace(x_min, x_max, 1000)
xxm, yym = np.meshgrid(xx, yy)
for num in knn_values:
    new_model = KNeighborsRegressor(n_neighbors = num)
    new_model.fit(X, y)
    X_grid = np.c_[xxm.ravel(), yym.ravel()]
    pred = new_model.predict(X_grid)
    knn_results[num] = pred

#
# Site Stuff
#

st.markdown("<h1 style=\"font-family: 'Helvetica'; font-size: 3em; text-align: center;\">\
Well Tempered</h1>"
    ,unsafe_allow_html=True)
st.markdown("<h1 style=\"font-family: 'Helvetica'\">Well Locations</h1>"
    ,unsafe_allow_html=True)

feature_used = st.radio('Location:', location_features, index=0)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=training_data[feature_used + '_x'],
    y=training_data[feature_used + '_y'],
    mode='markers',
    marker=dict(
        color=training_data['OilPeakRate'],
        colorscale=px.colors.sequential.Agsunset,
        size=3,
        colorbar=dict(title='OilPeakRate')
    ),
    hoverinfo='text',
    hovertext=['id: ' + str(id) for id in training_data['pad_id']]
))



fig.update_layout(
    plot_bgcolor='#0f1116',  # Dark plot background
    paper_bgcolor='#0f1116',  # Dark around the plot
    font_color="white",  # White font for contrast
    xaxis=dict(
        range=[x_min, x_max],
        showgrid=False
    ),
    yaxis=dict(
        range=[y_min, y_max],
        showgrid=False
    )
)

st.plotly_chart(fig, use_container_width=True)

three_n = px.imshow(knn_results[3].reshape((1000, 1000)), x = xx, y = yy)
five_n = px.imshow(knn_results[5].reshape((1000, 1000)), x = xx, y = yy)
ten_n= px.imshow(knn_results[10].reshape((1000, 1000)), x = xx, y = yy)
fifteen_n = px.imshow(knn_results[15].reshape((1000, 1000)), x = xx, y = yy)

new_fig = make_subplots(rows=2, cols=2, subplot_titles=("3 Neighbors", "5 Neighbors", "10 Neighbors", "15 Neighbors"))

# Add the image plots to the subplots, adjusting row and col as needed
new_fig.add_trace(three_n.data[0], row=1, col=1)
new_fig.add_trace(five_n.data[0], row=1, col=2)
new_fig.add_trace(ten_n.data[0], row=2, col=1)
new_fig.add_trace(fifteen_n.data[0], row=2, col=2)

##TODO setting this color scale doesn't do anything for some reason
new_fig.update_traces(autocolorscale = True, colorscale='viridis')

new_fig.update_layout(autosize=True, height=600, width=600)


st.plotly_chart(new_fig, use_container_width=True)