

# Customer Churn Prediction using RandomForestClassifier

This project predicts customer churn using the Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). The model is built using a Random Forest classifier and evaluated with various metrics like accuracy, confusion matrix, and ROC-AUC score.

## Dataset

The dataset used in this project is from the Telco Customer Churn dataset. It contains various features about customers such as their contract details, internet service usage, monthly charges, and whether they churned or not.

## Project Steps

1. **Data Preprocessing**:
   - **Handling Missing Values**: The `TotalCharges` column is converted to numeric, and rows with missing values in this column are dropped.
   - **Label Encoding**: Categorical variables are encoded using `LabelEncoder` to convert them into numerical format.
   - **Scaling**: Features are scaled using `StandardScaler` to ensure the model is not biased by differences in scale across features.

2. **Exploratory Data Analysis**:
   - **Visualization**: Several visualizations such as churn distribution, contract type vs. churn, and internet service type vs. churn are plotted to explore the dataset.
   - **Numerical Features**: Distribution of numerical features like `tenure`, `MonthlyCharges`, and `TotalCharges` is visualized using histograms.

3. **Model Building**:
   - **Train-Test Split**: The data is split into training and test sets using a 70-30 split.
   - **RandomForestClassifier**: A Random Forest model with 100 estimators is trained on the scaled training data.
   - **Prediction**: The trained model is used to make predictions on the test set.

4. **Evaluation**:
   - **Confusion Matrix**: The confusion matrix is generated to evaluate the performance of the model.
   - **Accuracy**: The accuracy of the model is calculated.
   - **Classification Report**: Precision, recall, F1-score, and support are presented.
   - **ROC-AUC Score**: The ROC-AUC score is calculated to assess the model's ability to distinguish between churn and non-churn customers.
   - **Feature Importance**: The importance of each feature in predicting churn is plotted.

## Installation

1. Clone the repository or download the script.
2. Install the required libraries using:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3. Download the Telco Customer Churn dataset from [here](https://www.kaggle.com/blastchar/telco-customer-churn) or use your own data file.

## Running the Project

1. Place the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) in the project directory.
2. Run the script in your Python environment (Google Colab, Jupyter Notebook, etc.).
3. The script will:
   - Preprocess the data.
   - Visualize key features.
   - Train the Random Forest model.
   - Evaluate the model's performance.
   - Display feature importances and confusion matrix visualizations.

## Project Output

- **Confusion Matrix**: A heatmap showing the model's prediction vs. actual results.
- **Accuracy**: The overall accuracy of the model.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **ROC-AUC Score**: An indication of the model's performance in predicting churn.
- **Feature Importance**: A bar plot showing the importance of each feature in the model.

## Acknowledgments

- The dataset is provided by [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).
- This project uses Scikit-learn's Random Forest Classifier to build the model.

