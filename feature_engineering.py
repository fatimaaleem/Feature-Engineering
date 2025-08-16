# 1. Print an explanation of what feature engineering means.
print("--- What is Feature Engineering? ---")
print("Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work.")
print("It involves transforming raw data into a format that is more suitable and informative for the model to learn from.")
print("Essentially, it's about creating new input features or transforming existing ones to improve the performance of a model.")

# 2. Print a summary explaining why feature engineering is important.
print("\n--- Why is Feature Engineering Important? ---")
print("Feature engineering is crucial because the success of a machine learning model heavily depends on the quality and relevance of the input features.")
print("Well-engineered features can:")
print("- Improve model accuracy and performance.")
print("- Help algorithms converge faster.")
print("- Lead to more interpretable models.")
print("- Reduce the need for complex models by making patterns more explicit.")
print("- Handle issues like missing values, outliers, and skewed distributions.")

# 3. Print an outline of the typical steps involved in the feature engineering process.
print("\n--- General Process of Feature Engineering ---")
print("The typical steps involved in feature engineering include:")
print("1. **Understanding the Data:** Deeply analyze the raw data, its structure, types, and potential issues (missing values, outliers, etc.). Understand the domain and the problem you are trying to solve.")
print("2. **Brainstorming Features:** Based on your understanding and domain knowledge, brainstorm potential new features that could be relevant to the model.")
print("3. **Creating Features:** Implement the techniques to create the new features or transform existing ones (e.g., handling missing values, encoding categorical data, scaling numerical data, extracting information from text or dates).")
print("4. **Evaluating Features:** Assess the quality and potential impact of the newly created or transformed features. This can involve visualization, statistical tests, or evaluating model performance with and without the features.")
print("5. **Selecting Features:** Choose the most relevant and impactful features for the model, potentially using feature selection techniques to reduce dimensionality and noise.")
print("6. **Iterating:** Feature engineering is often an iterative process. You might need to go back to earlier steps based on the evaluation results.")
import pandas as pd
import numpy as np  # Needed for np.nan

# 1. Print a brief explanation of why handling missing values is important.
print("--- Handling Missing Data ---")
print("\nMissing data is a common issue in real-world datasets and can significantly impact the performance of machine learning models. Many algorithms cannot handle missing values directly, and even those that can might produce biased or inaccurate results.")
print("Properly handling missing data is crucial to maintain data integrity, avoid errors during model training, and ensure that the model learns from a complete and representative dataset.")

# 2. Discuss common techniques for handling missing values.
print("\nCommon techniques for handling missing values include:")
print("- Deletion: Removing rows (listwise deletion) or columns (column-wise deletion) that contain missing values.")
print("- Imputation: Filling in missing values with estimated values. For example: mean, median, mode, KNN imputation, or model-based imputation.")

# 3. Create a sample pandas DataFrame with some missing values (NaN).
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, np.nan, 8, 9, 10],
    'C': ['X', 'Y', 'X', np.nan, 'Z'],
    'D': [np.nan, 12, 13, 14, 15]
}
df_missing = pd.DataFrame(data)
print("\n--- Original DataFrame with Missing Values ---")
print(df_missing)

# 4. Drop rows with missing values
print("\n--- DataFrame after dropping rows with any missing values ---")
df_dropped_rows = df_missing.dropna()
print(df_dropped_rows)

# 5. Impute missing values in 'A' with the mean
print("\n--- DataFrame after imputing missing values in column 'A' with the mean ---")
df_imputed_mean = df_missing.copy()
mean_A = df_imputed_mean['A'].mean()
df_imputed_mean['A'] = df_imputed_mean['A'].fillna(mean_A)
print(df_imputed_mean)

# 6. Impute missing values in 'C' with the mode
print("\n--- DataFrame after imputing missing values in column 'C' with the mode ---")
df_imputed_mode = df_missing.copy()
mode_C = df_imputed_mode['C'].mode()[0]  # Take the first mode
df_imputed_mode['C'] = df_imputed_mode['C'].fillna(mode_C)
print(df_imputed_mode)
# 1. Print an explanation of why transforming numerical features is important.
print("--- Transforming Numerical Features ---")
print("\nNumerical features often have varying scales, distributions, and relationships that can negatively impact the performance and training speed of many machine learning algorithms.")
print("Transforming numerical features helps to:")
print("- Standardize or normalize the range of values, preventing features with larger values from dominating the learning process (important for distance-based algorithms like K-Means, SVMs, and algorithms that use gradient descent).")
print("- Make the distribution of features more Gaussian-like, which can improve the performance of models that assume normally distributed inputs (e.g., linear regression, logistic regression).")
print("- Capture non-linear relationships between features and the target variable (e.g., using polynomial features).")
print("- Handle skewed distributions.")

# 2. Discuss common numerical feature transformation techniques.
print("\nCommon techniques for transforming numerical features include:")
print("- **Scaling:** Rescaling features to a specific range (e.g., [0, 1] or [-1, 1]) or to have a mean of 0 and a standard deviation of 1.")
print("  - **Purpose:** To ensure that all features contribute equally to the model, especially when features have vastly different scales.")
print("- **Normalization:** Transforming features to have a Gaussian-like distribution. This often involves non-linear transformations like logarithmic, square root, or Box-Cox transformations.")
print("  - **Purpose:** To satisfy the normality assumptions of certain models and reduce the impact of outliers.")
print("- **Binning (Discretization):** Converting continuous numerical features into discrete categories or bins.")
print("  - **Purpose:** To handle outliers, reduce the number of unique values (simplifying the model), or capture non-linear relationships by treating different ranges of values as distinct categories.")
print("- **Polynomial Features:** Creating new features by raising existing features to a power or combining them through multiplication.")
print("  - **Purpose:** To introduce non-linearity into the model and capture more complex relationships between features and the target.")
# 3. Import the necessary classes from sklearn.preprocessing: MinMaxScaler, StandardScaler, and PolynomialFeatures.
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
# Also, import numpy for creating sample data and pandas for easier handling and binning demonstration.
import numpy as np
import pandas as pd

# 4. Create a sample numpy array or pandas DataFrame with numerical data.
# Include data with different scales and potentially a skewed distribution.
data_numerical = {
    'Feature1_SmallScale': np.random.rand(20) * 10,  # Range ~ 0-10
    'Feature2_LargeScale': np.random.rand(20) * 1000 + 500,  # Range ~ 500-1500
    'Feature3_Skewed': np.random.chisquare(df=5, size=20) * 50,  # Skewed distribution
    'Feature4_Uniform': np.random.uniform(low=100, high=200, size=20)  # Uniform distribution
}

df_numerical = pd.DataFrame(data_numerical)

print("\n--- Original Numerical DataFrame ---")
print(df_numerical)  # use print() instead of display()
# 5. Demonstrate Min-Max Scaling using MinMaxScaler.
print("\n--- DataFrame after Min-Max Scaling ('Feature2_LargeScale') ---")
minmax_scaler = MinMaxScaler()
# Apply to a single column for demonstration, reshaping to a 2D array
df_numerical['Feature2_LargeScale_Scaled_MinMax'] = minmax_scaler.fit_transform(df_numerical[['Feature2_LargeScale']])
print(df_numerical[['Feature2_LargeScale', 'Feature2_LargeScale_Scaled_MinMax']].head())

# 6. Demonstrate Standardization (Z-score normalization) using StandardScaler.
print("\n--- DataFrame after Standardization ('Feature1_SmallScale') ---")
standard_scaler = StandardScaler()
# Apply to a single column for demonstration, reshaping to a 2D array
df_numerical['Feature1_SmallScale_Scaled_Standard'] = standard_scaler.fit_transform(df_numerical[['Feature1_SmallScale']])
print(df_numerical[['Feature1_SmallScale', 'Feature1_SmallScale_Scaled_Standard']].head())
