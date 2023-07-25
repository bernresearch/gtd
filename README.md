# Global Terrorism Database Analysis and Classification

### Overview
This project aims to analyze and classify terrorist attacks using the Global Terrorism Database (GTD) dataset. The objective is to build a machine learning model that can predict the type of attack based on various features associated with each incident.

### Data Source
The Global Terrorism Database (GTD) is a comprehensive open-source database that contains information on terrorist attacks worldwide. The dataset used in this project can be found on Kaggle: Global Terrorism Database on Kaggle.

### Project Steps
Data Loading and Preprocessing: The dataset is loaded into a pandas DataFrame, and irrelevant columns are dropped. Missing values are handled using imputation, and categorical columns are one-hot encoded for machine learning compatibility.

Data Splitting: The data is split into training and testing sets to train and evaluate the machine learning model.

Model Selection: A Random Forest classifier is chosen as the machine learning model due to its effectiveness in handling categorical data and preventing overfitting.

Model Training and Evaluation: The Random Forest classifier is trained on the training data and evaluated on the testing data using accuracy and classification report metrics.

### Results
The model shows an impressive accuracy of 99.95% on classifying attack types. However, it's important to note that the recall for the "True" class is relatively low, indicating room for improvement.

### Conclusion
This project demonstrates the application of machine learning techniques to analyze and classify terrorist attacks based on the Global Terrorism Database. The trained Random Forest classifier shows promising results in predicting attack types with high accuracy. Further exploration and refinement of the model could enhance its ability to correctly identify rare attack types.

### Future Work
Experiment with other machine learning algorithms to compare performance.
Conduct feature engineering to improve model performance and interpretability.
Collect additional data or explore external datasets to enrich the analysis.
Investigate the reasons behind the relatively low recall for the "True" class and address potential class imbalance.
