import streamlit as st
from PIL import Image

def app():
    st.title("Exploratory Data Analysis Page")

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

if __name__ == "__main__":
    app()