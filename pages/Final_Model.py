import streamlit as st
from PIL import Image


def app():
    st.title('Final Model and Analysis')
    st.markdown('## Initial Approaches')
    st.markdown('We attempted quite a few different approaches to this problem, including:')
    st.markdown('* Kernel Ridge Regression (RBF and Linear Kernels)')
    st.markdown('* Random Forest Regression')
    st.markdown('* KNN Regression')
    st.markdown('* Decision Tree Regression')
    st.markdown('* Gradient Boosting Regression')
    st.markdown('All of which greatly overfit or underfit the data, ultimately resulting in poor performance on the test set.')

    st.markdown('## Final Model')
    st.markdown('We finally chose to use AdaBoost Regression with a Histogram-based Gradient Boosting Tree as the base estimator.')
    st.markdown('The Histogram-based Gradient Boosting Tree is much faster than the traditional Gradient Boosting Tree, and is able to handle handles missing values elegantly.')
    st.markdown('The AdaBoost Regression algorithm improves the base estimator by forcing the base estimator to concentrate on harder cases as training goes on. ')
    st.markdown('On 5 different train-test splits, we achieved an average RMSE of `78.698` on the test set and `21.264` on the training set.')

    st.markdown('After verifying the validation performance, we trained the final model on the entire dataset.')
    st.code("""# Train final KNN model for K = 15
KNN_model = KNeighborsRegressor(n_neighbors=15)

# Fit the model to all the training data
KNN_model.fit(X, y)

y_pred = KNN_model.predict(X)

X['KNN_OilPeakRate'] = y_pred

# Train final AdaBoost model
final_model = AdaBoostRegressor(estimator=HistGradientBoostingRegressor(max_iter=1000, l2_regularization=0.1),
                          n_estimators=15)

final_model.fit(X, y)

# Save the model to a pickle file
pickle.dump(KNN_model, open('Model/KNN_final.pkl', 'wb'))

# Save the encoder to a pickle file
pickle.dump(encoder, open('Model/encoder.pkl', 'wb'))

# Save the model to a pickle file
pickle.dump(final_model, open('Model/final_model.pkl', 'wb'))""", language='python')

    st.markdown('## Interpretation')
    st.markdown("Because AdaBoost fits an ensemble of regressors, it can be difficult to interpret. As a proxy, we fit and analyze an ordinary decision tree.")
    st.markdown("The feature importance plot shows that our KNN-based spatial figure is by far the most important, followed by `total_proppant` and other physical features.")

    featureimportance_image = Image.open('Figures/featureimportance.png')
    st.image(featureimportance_image, caption='Feature Importance', width=1000)

    st.markdown("We also visualize the first three layers of the decision tree, which demonstrate the importance of the KNN-based feature and `total_proppant`.")
    decision_tree_image = Image.open('Figures/decision_tree.png')
    st.image(decision_tree_image, caption='Truncated Decision Tree', width=1000)


if __name__ == "__main__":
    app()