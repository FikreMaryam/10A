import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load datasets
data = pd.read_csv('../files/benin-malanville.csv')
data2 = pd.read_csv('../files/sierraleone-bumbuna.csv')
data3 = pd.read_csv('../files/togo-dapaong_qc.csv')

# List of columns to calculate statistics for
columns = ['RH', 'WS', 'GHI', 'DNI', 'DHI']  # Replace with your column names
statistics = {}
for col in columns:
    if col in data.columns:
        statistics[col] = {
            'Mean': data[col].mean(),
            'Median': data[col].median(),
            'Variance': data[col].var(),
            'Skewness': data[col].skew(),
            'Kurtosis': data[col].kurt()
        }

stats_df = pd.DataFrame(statistics).transpose()
print("Summary Statistics:\n", stats_df)

# -------------------------------
# Data Quality Check
# -------------------------------

# 1. Missing Values - Create a table format with missing values and percentage
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100
missing_data = pd.DataFrame({
    'Missing Values Count': missing_values,
    'Missing Percentage (%)': missing_percentage
})

# Filter out columns with no missing values and round percentage for clarity
missing_data = missing_data[missing_data['Missing Values Count'] > 0]
missing_data['Missing Percentage (%)'] = missing_data['Missing Percentage (%)'].round(2)

# Print the table in a more understandable format
print("\nMissing Values in Columns (Count & Percentage):\n")
print(missing_data.to_string())

# 2. Identifying Columns and Rows with Missing Values for each column
for col in missing_values[missing_values > 0].index:
    missing_rows = data[data[col].isnull()].index.tolist()
    print(f"\nColumn '{col}' has missing values in the following rows:")
    print(missing_rows)

# 3. Negative Values
columns_to_check_negative = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
for col in columns_to_check_negative:
    if col in data.columns:
        negative_count = (data[col] < 0).sum()
        print(f"Negative values in {col}: {negative_count}")

# 4. Outliers (Using Z-Score)
numeric_columns = data.select_dtypes(include=[np.number]).columns
z_scores = data[numeric_columns].apply(zscore)

# Outlier detection (Z-score > 3)
outliers = (z_scores.abs() > 3).sum()
print("\nOutliers Detected in Numeric Columns:\n", outliers)

# 5. Visualization of Outliers (Boxplot with description)
plt.figure(figsize=(12, 8))
sns.boxplot(data=data[columns_to_check_negative])
plt.title("Boxplot for Outlier Detection")
plt.xticks(rotation=45)
plt.xlabel("Variables")
plt.ylabel("Values")
plt.grid(True)
plt.text(0.5, 150, 'Outliers are values significantly higher or lower than the rest of the data.',
         fontsize=12, color='red', ha='center', va='center')
plt.show()

# -------------------------------
# Additional Data Quality Visualization
# -------------------------------

# Histogram for numerical columns to observe data distribution
data[columns].hist(bins=30, figsize=(12, 8), grid=True)
plt.suptitle("Histograms for GHI, DNI, DHI, RH, WS")
plt.show()

plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
plt.plot(data['Timestamp'], data['GHI'], label='GHI', color='orange')
plt.title('GHI over Time')
plt.xlabel('Time')
plt.ylabel('GHI')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(data['Timestamp'], data['DNI'], label='DNI', color='blue')
plt.title('DNI over Time')
plt.xlabel('Time')
plt.ylabel('DNI')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(data['Timestamp'], data['DHI'], label='DHI', color='green')
plt.title('DHI over Time')
plt.xlabel('Time')
plt.ylabel('DHI')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(data['Timestamp'], data['Tamb'], label='Tamb', color='red')
plt.title('Tamb (Temperature) over Time')
plt.xlabel('Time')
plt.ylabel('Tamb (°C)')
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# Monthly Analysis of GHI, DNI, DHI, and Tamb
# -------------------------------

# Set Timestamp as index and resample by month
data.set_index('Timestamp', inplace=True)

monthly_data = data.resample('M').mean()  # Resample by month and get the mean for each month

plt.figure(figsize=(14, 8))

# Bar charts for monthly average GHI, DNI, DHI, and Tamb
plt.subplot(2, 2, 1)
plt.bar(monthly_data.index, monthly_data['GHI'], label='GHI', color='orange')
plt.title('Monthly GHI')
plt.xlabel('Month')
plt.ylabel('GHI')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.bar(monthly_data.index, monthly_data['DNI'], label='DNI', color='blue')
plt.title('Monthly DNI')
plt.xlabel('Month')
plt.ylabel('DNI')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.bar(monthly_data.index, monthly_data['DHI'], label='DHI', color='green')
plt.title('Monthly DHI')
plt.xlabel('Month')
plt.ylabel('DHI')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.bar(monthly_data.index, monthly_data['Tamb'], label='Tamb', color='red')
plt.title('Monthly Tamb (Temperature)')
plt.xlabel('Month')
plt.ylabel('Tamb (°C)')
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# Impact of Cleaning on ModA, ModB (Using the 'Cleaning' Column)
# -------------------------------

# Assuming 'Cleaning' column contains binary values, 1 for cleaned and 0 for not cleaned
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
sns.boxplot(x='Cleaning', y='ModA', data=data, palette='Set2')
plt.title('ModA Sensor Readings vs Cleaning Status')
plt.xlabel('Cleaning Status')
plt.ylabel('ModA Sensor Readings')

plt.subplot(2, 2, 2)
sns.boxplot(x='Cleaning', y='ModB', data=data, palette='Set2')
plt.title('ModB Sensor Readings vs Cleaning Status')
plt.xlabel('Cleaning Status')
plt.ylabel('ModB Sensor Readings')

plt.tight_layout()
plt.show()

# -------------------------------
# Additional Time Series Analysis: ModA and ModB over time
# -------------------------------

# Plot ModA and ModB over time to check trends
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['ModA'], label='ModA', color='purple')
plt.title('ModA Sensor Readings over Time')
plt.xlabel('Time')
plt.ylabel('ModA')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(data.index, data['ModB'], label='ModB', color='brown')
plt.title('ModB Sensor Readings over Time')
plt.xlabel('Time')
plt.ylabel('ModB')
plt.grid(True)

plt.tight_layout()
plt.show()
