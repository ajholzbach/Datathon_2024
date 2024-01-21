import streamlit as st
from PIL import Image

def app():
    st.title("Data Wrangling")

    st.markdown("## Data Cleaning")
    st.markdown("First, We read in the provided training data, which has `x` numerical features and `y` categorical features.")

    st.markdown('Several features have "Undefined", "Unknown", or "NaN" values. We replace "Undefined" and "Unknown" values with "NaN" values, and show them in the figure below.')
    nan_image = Image.open('Figures/NaNs.png')
    st.image(nan_image, caption='NaN Values', width=1000)

    drop_columns = ['Unnamed: 0', 'pad_id', 'standardized_operator_name', 'average_stage_length', 'average_proppant_per_stage', 'average_frac_fluid_per_stage', 'number_of_stages', 'frac_type']
    st.markdown("We drop several columns which have meaningless or mostly null values:")
    st.markdown(" * `Unnamed: 0` is the index value and is removed immediately")
    st.markdown(" * `standardized_operator_name` is a bookkeeping value defining operators")
    st.markdown(" * `pad_id` is an identifier for the pad, and is not useful for our purposes")
    st.markdown(" * `number_of_stages`, `average_stage_length`, `average_proppant_per_stage`, and `average_frac_fluid_per_stage` are mostly null")
    st.markdown(" * `frac_type` has categories with 0 or 1 members in the whole dataset")
    
    st.code("""# Load the data
training_data = pd.read_csv('Data/training.csv')
drop_columns = ['Unnamed: 0', 'pad_id', 'standardized_operator_name', 'average_stage_length', 'average_proppant_per_stage', 'average_frac_fluid_per_stage', 'number_of_stages', 'frac_type']
training_data.drop(columns=drop_columns, inplace=True)
training_data.drop_duplicates(inplace=True)
training_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Convert any "Undefined" or "Unknown" values to NaN
training_data.replace('Undefined', np.nan, inplace=True)
training_data.replace('Unknown', np.nan, inplace=True)""", language='python')

    st.markdown("## Data Imputation")
    st.markdown("Before imputing, we save the rows with missing values in the target variables for later removal.")
    st.markdown(" * For categorical variables, we impute missing values with a 'Missing' category and then encode them with an Ordinal Encoder.")
    st.markdown(" * For numerical variables, we impute missing values using KNN imputation with 10 nearest neighbors.")
    st.code("""# Before imputing, keep track of rows with missing values for 'OilPeakRate' to remove them later
rows_with_missing_values = training_data[training_data['OilPeakRate'].isnull()].index.to_list()

# Label categorical variables
categorical_columns = ['ffs_frac_type', 'relative_well_position', 'batch_frac_classification', 'well_family_relationship']

# Get numeric columns
numeric_columns = training_data.select_dtypes(include=np.number).columns.to_list()

# Impute categorical variables with the most frequent value
imputer = SimpleImputer(strategy='constant', fill_value='missing')
training_data[categorical_columns] = imputer.fit_transform(training_data[categorical_columns])

# Impute numerical variables with the KNN algorithm
imputer = KNNImputer(n_neighbors=10)
training_data[numeric_columns] = imputer.fit_transform(training_data[numeric_columns])

# Encode categorical variables using OrdinalEncoder
encoder = OrdinalEncoder()
training_data[categorical_columns] = encoder.fit_transform(training_data[categorical_columns])""", language='python')

    st.markdown("## Outlier Removal")
    st.markdown("We start by removing all rows that originally had missing values in the target variables.")
    st.markdown("We then remove outliers in the target variables by calculating the interquartile range and removing all rows with values outside of 1.5 times the interquartile range.")
    st.code("""# Drop rows with missing values for 'OilPeakRate'
training_data.drop(index=rows_with_missing_values, inplace=True)

training_data_copy = training_data.copy()

# Remove outliers in 'OilPeakRate' based on the 1.5*IQR rule
Q1 = training_data['OilPeakRate'].quantile(0.25)
Q3 = training_data['OilPeakRate'].quantile(0.75)
IQR = Q3 - Q1
print('Q1: ', Q1)
print('Q3: ', Q3)
print('IQR: ', IQR)
upper_bound = Q3 + 1.5*IQR
lower_bound = Q1 - 1.5*IQR
training_data = training_data[(training_data['OilPeakRate'] < upper_bound) & (training_data['OilPeakRate'] > lower_bound)]""", language='python')

if __name__ == "__main__":
    app()