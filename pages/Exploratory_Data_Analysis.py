import streamlit as st
from PIL import Image
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def app():
    st.title("Exploratory Data Analysis")

    st.markdown("## Feature Correlation")
    st.markdown("We start by looking at the correlations between all of the features and each other, as well as the target variables.")
    correlation_heatmap = Image.open('Figures/correlation_heatmap.png')
    st.image(correlation_heatmap, caption='Correlation Heatmap', width=1000)
    st.markdown("As you can see, some of the features are highly correlated with each other and might be redundant.")

    st.markdown("## Categorical Trends")

    st.markdown("To look for initial trends, we plot boxplots of the categorical features, divided by category. Aside from outliers, the response is similarly distributed across categories. This explains why the importance of categorical variables is low in our final model.")
    col1, col2 = st.columns(2)
    with col1:
        relative_well_position_image = Image.open('Figures/relative_well_position_boxplot.png')
        st.image(relative_well_position_image, caption='Relative Well Position Boxplot', width=600)
        batch_frac_classification_image = Image.open('Figures/batch_frac_classification_boxplot.png')
        st.image(batch_frac_classification_image, caption='Batch Frac Classification Boxplot', width=600)
    with col2:
        ffs_frac_type_image = Image.open('Figures/ffs_frac_type_boxplot.png')
        st.image(ffs_frac_type_image, caption='FFS Frac Type Boxplot', width=600)
        well_family_relationship_image = Image.open('Figures/well_family_relationship_boxplot.png')
        st.image(well_family_relationship_image, caption='Well Family Relationship Boxplot', width=600)

    st.markdown("## Spatial Analysis")

    st.markdown("We noticed that peak oil production exibhits spatial trends, high oil production is concentrated in a few areas of the map.")
    spatial_image = Image.open('Figures/facial_info.png')
    st.image(spatial_image, caption='Spatial Distribution of Peak Oil Production')

    st.markdown("## Feature Engineering")
    st.markdown("Based on the spatial data, we hypothesized that the best predictor of peak oil production is the perfomance of wells around the target well.")
    st.markdown("In order to capture this, we created a KNN model to predict the peak oil production of a well base on the peak oil production of its neighbors. The output of this model is then fed as a feature into our final model.")

    # Make a slider to control the number of neighbors (Either 2, 8, 15, 30, or 50)
    kval = st.slider("Number of Neighbors", min_value=2, max_value=30, step=7)

    # Load predictions from pickle files
    predictions = {}
    for k in [2, 9, 16, 23, 30]:
        with open(f'KNN/y_pred_{k}.pkl', 'rb') as f:
            predictions[k] = pickle.load(f)
    # Load meshgrid from pickle file
    with open('KNN/xx.pkl', 'rb') as f:
        xx = pickle.load(f)
    with open('KNN/yy.pkl', 'rb') as f:
        yy = pickle.load(f)

    # Plot the meshgrid with plotly
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Heatmap(z=predictions[kval].reshape((1000, 1000)), showscale=True, colorscale="viridis"), row=1, col=1)
    # Update the layout
    fig.update_layout(
        plot_bgcolor='#0f1116',  # Dark plot background
        paper_bgcolor='#0f1116',  # Dark around the plot
        font_color="white",  # White font for contrast
        xaxis=dict(
            range=[0, 1000],
            showgrid=False
        ),
        yaxis=dict(
            range=[0, 1000],
            showgrid=False
        ),
        autosize=False,
        width=1000,
        height=1000
    )
    # Make tooltips show the x and y coordinates as well as the OilPeakRate
    fig.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>OilPeakRate: %{z}")
    # Display the figure
    st.plotly_chart(fig, use_container_width=False)

    st.markdown("We chose a K value of 15 for our final model, as it seemed to capture the spatial trends without much overfitting.")

    st.markdown("Besides the spatial data, we included two other engineered features in our final model:")
    st.markdown(" * The first is the birds-eye distance between the surface location of the well and the bottom hole.")
    st.code("""# Create a 'well_skew' feature
# The formula for well skew is sqrt(|surface_x - bh_x|^2 + |surface_y - bh_y|^2)
training_data['well_skew'] = np.sqrt((training_data['surface_x'] - training_data['bh_x'])**2 + (training_data['surface_y'] - training_data['bh_y'])**2)""", language='python')
    st.markdown(" * The second is the estimated lateral length of the well, which is the distance between the bottom hole and the lateral toe.")
    st.code("""# Create a 'lateral_length' feature
# The formula for lateral length is sqrt(|horizontal_toe_x - bh_x|^2 + |horizontal_toe_y - bh_y|^2)
training_data['lateral_length'] = np.sqrt((training_data['horizontal_toe_x'] - training_data['bh_x'])**2 + (training_data['horizontal_toe_y'] - training_data['bh_y'])**2)""", language='python')

if __name__ == "__main__":
    app()