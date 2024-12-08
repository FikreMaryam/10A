import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('../files/benin-malanville.csv')
data2 = pd.read_csv('../files/sierraleone-bumbuna.csv')
data3 = pd.read_csv('../files/togo-dapaong_qc.csv')

# Example: Calculating statistics for the 'RH' column
mean_value = data['RH'].mean()
median_value = data['RH', 'WS'].median()
variance_value = data['RH'].var()
skewness_value = data['RH'].skew()
kurtosis_value = data['RH'].kurt()

# Create a dictionary to organize the statistics
stats = {
    "Statistic": ["Mean", "Median", "Variance", "Skewness", "Kurtosis"],
    "Value": [mean_value, median_value, variance_value, skewness_value, kurtosis_value]
}
# Convert the dictionary into a DataFrame and Print the DataFrame
stats_df = pd.DataFrame(stats) 
print(stats_df)

# 1. Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# 2. Check for negative values where they shouldn't exist (e.g., GHI, DNI, DHI)
negative_values_check = data[['GHI', 'DNI', 'DHI']].lt(0).sum()
print("\nNegative values check for GHI, DNI, DHI:\n", negative_values_check)

# 3. Check for outliers using IQR (Interquartile Range) for relevant columns
def detect_outliers_iqr(df, RH):
    Q1 = df[RH].quantile(0.25)
    Q3 = df[RH].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[RH] < lower_bound) | (df[RH] > upper_bound)]
    return outliers

# Check for outliers in ModA, ModB, WS, and WSgust
outliers_ModA = detect_outliers_iqr(data, 'ModA')
outliers_ModB = detect_outliers_iqr(data, 'ModB')
outliers_WS = detect_outliers_iqr(data, 'WS')
outliers_WSgust = detect_outliers_iqr(data, 'WSgust')

print("\nOutliers in ModA:\n", outliers_ModA)
print("\nOutliers in ModB:\n", outliers_ModB)
print("\nOutliers in WS:\n", outliers_WS)
print("\nOutliers in WSgust:\n", outliers_WSgust)

# 4. Visualize the data using boxplots to identify outliers visually
sns.boxplot(data['GHI'])
plt.title('Boxplot for GHI')
plt.show()

sns.boxplot(data['DNI'])
plt.title('Boxplot for DNI')
plt.show()

sns.boxplot(data['DHI'])
plt.title('Boxplot for DHI')
plt.show()

sns.boxplot(data['ModA'])
plt.title('Boxplot for ModA')
plt.show()

sns.boxplot(data['ModB'])
plt.title('Boxplot for ModB')
plt.show()

sns.boxplot(data['WS'])
plt.title('Boxplot for WS')
plt.show()

sns.boxplot(data['WSgust'])
plt.title('Boxplot for WSgust')
plt.show()
