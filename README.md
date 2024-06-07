![Heart Disease Image](https://beaumonteh.com/wp-content/uploads/2021/12/Heart-Disease-CVD-Cardiovascular-Disease.jpg)

# Heart Disease Prediction App built by using KNN Model

## Context

Heart disease remains one of the leading causes of mortality worldwide, accounting for a significant number of deaths annually. According to WHO, this disease causes up to 17.9 M global mortality on an annual basis. Therefore, timely diagnosis and monitoring are crucial for managing heart conditions effectively and reducing associated risks. In recent years, advancements in machine learning techniques have shown promise in assisting healthcare professionals with accurate prediction and diagnosis of heart diseases.

## Objective

The primary objective of this capstone project is to develop a user-friendly heart prediction application powered by machine learning algorithms. The application aims to provide individuals with a convenient tool to assess their risk of developing heart diseases based on various medical and lifestyle factors. By leveraging machine learning models trained on comprehensive datasets (from [UCI ML](https://archive.ics.uci.edu/dataset/45/heart+disease)), the app intends to offer personalized risk assessments of heart diseases based on some medical features.

## Analytical approach

We want to analyze data to learn about patterns from features that can differentiate which patients are more likely to get heart disease and who will not. Then, we will build a binary classification model to help medical staff based on the analyzed data.

## Metric Evaluation

We want to focus on patients who have heart diseases, so we decided our target shown below as

Target:
* 0 : Patients who do not have heart diseases
* 1 : Patients who do have heart diseases

<img src="https://assets-global.website-files.com/6266b596eef18c1931f938f9/644aea65cefe35380f198a5a_class_guide_cm08.png" alt="Confusion Matrix" width="1000">

Based on the model, we will have two kinds of errors
Type 1 error: False Positive (Patients are predicted as having heart diseases. However in actual conditions, they do not have heart diseases)
Consequences: The patient need to conduct further medical examination but only for assessment or verification to confirm. Due to the mistake, the image of the hospital & app developer became less reliable to the patients and the public.

Type 2 error: False Negative (Patients are predicted as not having heart diseases. However in actual conditions, they do have heart diseases)
Consequences: Patient conditions may become worse. If they do manage to get diagnosed, their heart disease treatments are more likely to be more difficult and far more expensive. Even, they may have a probability of dying before being treated.

Based on the consequences above, we need to be able to make a model that can reduce False Negatives as much as possible because the consequences of False Negatives are higher and we may lose a person's life. Thus, we are using recall as a metric of evaluation.

## Conclusion
After training and testing 9 machine learning models (Random Forest, Logistic Regression, KNN, CatBoost, Gradient Boost, AdaBoost, Decision tree, LightGBM, XGBoost) to the dataset, the tuned KNN is the best model for predicting heart disease in this scenario, due to its high recall of 0.891, F1-score of 0.86 and best ROC AUC score of 0.85.
