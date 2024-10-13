# Case-2-Predicting-Student-Success-in-Online-Courses-
### Documentation for Student Completion Prediction Model

---

#### Overview

This project involves building a machine learning model to predict whether a student is likely to complete their courses based on their engagement and historical data. The model uses multiple classification algorithms and performs data preprocessing, feature engineering, and hyperparameter tuning.

---

### 1. **Data Preprocessing**

Three datasets are merged to create a single dataset that contains student profiles, course engagement metrics, and historical academic data. 

#### Steps:
- **Loading the data**: The three datasets are loaded using `pandas`.
  ```python
  profile_df = pd.read_csv('student_profile_data.csv')
  engagement_df = pd.read_csv('course_engagement_data.csv')
  historical_df = pd.read_csv('historical_data.csv')
  ```
- **Merging the datasets**: All three datasets are merged based on the `student_id` column.
  ```python
  merged_df = pd.merge(profile_df, engagement_df, on='student_id')
  merged_df = pd.merge(merged_df, historical_df, on='student_id')
  ```

- **Handling missing values**: 
  - Numeric columns are filled with the mean of the column.
  - Categorical columns are filled with the mode.
  ```python
  merged_df['logins_per_week'].fillna(merged_df['logins_per_week'].mean(), inplace=True)
  merged_df['gender'].fillna(merged_df['gender'].mode()[0], inplace=True)
  ```

---

### 2. **Feature Engineering**

A target variable `completion_status` is created based on the threshold criteria for predicting whether a student will complete their course:

- `courses_completed > 3`
- `average_quiz_score > 70`
- `logins_per_week > 4`

The function to determine the completion status:
```python
def determine_completion_status(row):
    if (row['courses_completed'] > 3 and 
        row['avg_score_across_courses'] > 70 and 
        row['logins_per_week'] > 4):
        return 1  # Likely to complete
    else:
        return 0  # Likely to drop out
```

---

### 3. **Exploratory Data Analysis (EDA)**

- **Pairplots**: Visualizing relationships between features like `logins_per_week`, `avg_quiz_score`, `courses_completed`, and `avg_score_across_courses` using a pairplot.
  ```python
  sns.pairplot(merged_df, hue='completion_status')
  plt.show()
  ```

- **Correlation heatmap**: A correlation matrix is calculated and plotted to understand the correlation between different features.
  ```python
  sns.heatmap(encoded_df.corr(), annot=True)
  plt.show()
  ```

---

### 4. **Model Training and Evaluation**

Several classification algorithms were used, and hyperparameter tuning was performed on the Random Forest model using `GridSearchCV`.

#### Models Evaluated:
1. **Random Forest**
2. **Gradient Boosting**
3. **Logistic Regression**
4. **Support Vector Classifier (SVC)**

```python
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Classifier': SVC(probability=True)
}
```

#### Performance Metrics:
For each model, the following metrics are evaluated:
- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: The ability of the classifier to not label a negative class as positive.
- **Recall**: The ability to find all the positive samples.
- **F1-Score**: The harmonic mean of precision and recall.

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/8f812322-c4fd-48b5-8c79-f0491f0ee60a)
![image](https://github.com/user-attachments/assets/2e992d83-8f90-4600-bc23-d2ffbc3640c3)



#### Hyperparameter Tuning:
A grid search is used to optimize the hyperparameters of the Random Forest model:
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', verbose=1)
grid_search.fit(X_train, y_train)
```

---

### 5. **Model Evaluation and Finalization**

The best model from grid search is used to make final predictions and evaluate performance on the test set:
```python
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
```

The performance metrics of the final model are reported:
```python
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
```

---

### 6. **Model Export**

The best-performing model is saved using the `joblib` library for future use:
```python
import joblib
joblib.dump(best_model, 'student_completion_model.pkl')
```

---

### 7. **Recommendation Metrics (MAP@K and NDCG@K)**

To evaluate the ranking quality of predicted items, MAP@K and NDCG@K metrics are calculated.

- **MAP@K (Mean Average Precision at K)**: Measures how well the model ranks relevant items among the top K predictions.
  ```python
  def mapk(actual_list, predicted_list, k=10):
      return np.mean([apk(a, p, k) for a, p in zip(actual_list, predicted_list)])
  ```

- **NDCG@K (Normalized Discounted Cumulative Gain at K)**: Evaluates the ranking quality with more weight given to higher-ranked relevant items.
  ```python
  def ndcg_at_k(actual, predicted, k=10):
      ideal_dcg = dcg_at_k(sorted([1 if i in actual else 0 for i in actual], reverse=True), k)
      dcg = dcg_at_k(relevance_scores, k)
      return dcg / ideal_dcg
  ```

---

This documentation outlines the steps and code used for the student course completion prediction model, covering data processing, model training, evaluation, and final recommendations.
