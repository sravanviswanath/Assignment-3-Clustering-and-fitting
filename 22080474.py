#!/usr/bin/env python
# coding: utf-8

# # Library Importation

# In[1]:


# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy.stats import t


# In[2]:


# Read the dataset 
population_data = pd.read_csv("API_SP.URB.TOTL_DS2_en_csv_v2_6301061.csv", skiprows = 3)
population_data.head(5)


# In[3]:


# Explore patterns of the dataset
print(population_data.shape)
print(population_data.columns)   
print(population_data.describe())    


# In[4]:


population_data.isna().sum()   # Identify columns for null values


# In[5]:


# Drop column
population_data = population_data.drop(columns=['Unnamed: 67'])
columns_to_fill = population_data.columns[4:]
# Fill null values with the mean 
population_data[columns_to_fill] = population_data[columns_to_fill].apply(lambda col: col.fillna(col.mean()), axis=0)
population_data.isna().sum()


# In[6]:


# Selecting data for the year 2000
first_dataframe = population_data[['Country Name', 'Country Code', 'Indicator Name', '2000']].copy()
first_dataframe = first_dataframe.rename(columns={'2000': 'Urban Population'})
print("Dataframe for 2000:")
first_dataframe.head(10)


# In[7]:


# Selecting data for the year 2000
second_dataframe = population_data[['Country Name', 'Country Code', 'Indicator Name', '2022']].copy()
second_dataframe = second_dataframe.rename(columns={'2022': 'Urban Population'})
print("Dataframe for 2022:")
second_dataframe.head(10)


# In[8]:


# Merge the two dataframes 
merged_dataframe = pd.merge(first_dataframe, second_dataframe, on=['Country Name', 'Country Code', 'Indicator Name'], how='inner', suffixes=('_2000', '_2022'))
# Print the merged dataframe
print("Merged Dataframe:")
merged_dataframe.head(10)


# # K-means Clustering

# In[9]:


# Extract relevant columns for clustering
cluster_columns = ['Urban Population_2000', 'Urban Population_2022']
cluster_data = merged_dataframe[cluster_columns]

# Standardize the data 
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Apply k-means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)  
clusters = kmeans.fit_predict(cluster_data_scaled)

# Add cluster information to the dataframe
merged_dataframe['Cluster'] = clusters

# Define individual colors for each cluster
cluster_colors = ['brown', 'blue', 'green', 'yellow', 'orange']

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Plot the clusters using a scatter plot with individual colors
plt.figure(figsize=(6, 4))
scatter = plt.scatter(cluster_data_scaled[:, 0], cluster_data_scaled[:, 1], c=[cluster_colors[i] for i in clusters], edgecolors='k', s=50)

# Plot cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='o', s=150, label='Cluster Centers')

# Create legend handles and labels for each cluster
legend_handles = []
for i, color in enumerate(cluster_colors):
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Cluster {i}'))

# Add legend with specified handles and labels
plt.legend(handles=legend_handles + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Cluster Centers')], loc='upper left')

plt.title('K-Means Clustering of Urban Population in 2000 and 2022')
plt.xlabel('Urban Population in 2000')
plt.ylabel('Urban Population in 2022')
plt.show()


# # Curve Fitting

# In[10]:


# Function for exponential growth model
def exponential_growth(x, a, b, c):
    return a * np.exp(b * x) + c

# Sample data 
time_points = np.array([0, 1, 2, 3, 4, 5])
population = np.array([100, 120, 150, 180, 220, 250])

# Fit the exponential growth model to the data
params, covariance = curve_fit(exponential_growth, time_points, population)

# Estimated parameters
a, b, c = params

# Predict future values
future_time_points = np.array([6, 7, 8, 9, 10])
predicted_population = exponential_growth(future_time_points, a, b, c)

# Estimate confidence intervals using err_ranges
def err_ranges(func, x, popt, pcov, alpha=0.05):
    p = len(popt)  # Number of parameters
    dof = max(0, len(x) - p)  # Degrees of freedom
    t_value = abs(t.ppf(alpha / 2, dof))  # t-distribution value for confidence interval

    popt_err = np.sqrt(np.diag(pcov))  # Standard deviations of the parameters

    lower_bound = func(x, *popt - t_value * popt_err)
    upper_bound = func(x, *popt + t_value * popt_err)

    return lower_bound, upper_bound

# Estimate confidence intervals
lower_bound, upper_bound = err_ranges(exponential_growth, time_points, params, covariance)


# In[11]:


# Plot the data, best-fitting function, and confidence intervals
plt.scatter(time_points, population, label='Actual Data')
plt.plot(time_points, exponential_growth(time_points, a, b, c), label='Best Fitting Function', color='orange')
plt.fill_between(time_points, lower_bound, upper_bound, color='skyblue', alpha=0.2, label='Confidence Interval')

# Plot predicted future values
plt.plot(future_time_points, predicted_population, marker='o', linestyle='dashed', color='purple', label='Predicted Values')

plt.title('Exponential Growth Model and Predictions')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()


# # Investigate Trends in Clusters

# In[12]:


# Define the countries
countries = merged_dataframe['Country Code']

# Plot line plots for each cluster
plt.figure(figsize=(12, 8))

for cluster_num in range(5):  
    plt.subplot(2, 3, cluster_num + 1)

    # Select one country from each cluster
    country_in_cluster = merged_dataframe[merged_dataframe['Cluster'] == cluster_num].iloc[0]['Country Code']

    # Plot the data for the selected country
    for country in [country_in_cluster] + list(countries.sample(n=4)):  # Select 4 more random countries
        country_data = merged_dataframe.loc[merged_dataframe['Country Code'] == country, cluster_columns].values.flatten()
        plt.plot(cluster_columns, country_data, label=country)

    plt.title(f'Cluster {cluster_num}')
    plt.xlabel('Year')
    plt.ylabel('Urban Population')
    plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




