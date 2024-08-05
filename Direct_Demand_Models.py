#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import time
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from multiprocessing import Process, Manager, Semaphore
from joblib import Parallel, delayed
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import os
from tkinter import ttk, simpledialog, messagebox, font


# # Reading database

# In[2]:


# Get current working directory
current_directory = os.getcwd()

# Name training database
file_path = os.path.join(current_directory, 'training_dataset.csv')

# Read training database
df = pd.read_csv(file_path)

# Remove multicollinearity variables
df = df.drop(['vrm','ms_single','ms_married','ms_widowed','edu_graduate','modet_taxi','veh_three','veh_fourplus',
              'sex_male','age_64plus','income_200','income_200plus','race_indian',
              'race_other','race_hawaiian','race_mixed','race_white','modet_drive',
              'ms_divorced','ms_separated'], axis=1)


# ## Standardize the training and prediction datasets

# In[3]:


# Extract categorical variables
cluster_column = df['Cluster']
upt_column = df['upt']
month_column = df['month']
df = df.drop(['Cluster','upt','month'], axis=1)

# Fit the scaler on the training data
scaler = StandardScaler(with_mean = True, # center the data on mean = zero
                        with_std=True) # set standard deviation = 1
scaler.fit(df)

# Transform the training dataset
scaled_data = scaler.transform(df)

# Construct the database using the standarize data
df_names = df.columns.tolist()
df = pd.DataFrame(scaled_data, columns = df_names)
df['Cluster'] = cluster_column
df['upt'] = upt_column
df['month'] = month_column

# Transform variables to categorical variables
df['month'] = df['month'].astype('category')
df['Cluster'] = df['Cluster'].astype('category')


# In[4]:


# Read observations to predict
file_path = os.path.join(current_directory, 'prediction_dataset.csv')
x_prediction = pd.read_csv(file_path)

# Remove multicollinearity variables
x_prediction = x_prediction.drop(['vrm','ms_single','ms_married','ms_widowed','edu_graduate','modet_taxi','veh_three','veh_fourplus',
                                  'sex_male','age_64plus','income_200','income_200plus','race_indian',
                                  'race_other','race_hawaiian','race_mixed','race_white','modet_drive',
                                  'ms_divorced','ms_separated'], axis=1)

# Remove categorical variables
month_column_test = x_prediction['month']
x_prediction = x_prediction.drop(['month'], axis=1)

# Transform the testing data using the fitted scaler
x_prediction = scaler.transform(x_prediction)

# Construct the standardized testing dataset
x_prediction = pd.DataFrame(x_prediction, columns=df_names)
x_prediction['month'] = month_column_test

# Transform variables to categorical variables
x_prediction['month'] = x_prediction['month'].astype('category')


# # Request the cluster

# In[5]:


# Request the number of cluster to be analyzed
while True:
    cluster_number = simpledialog.askstring("Input", "Please enter the Cluster ID, including 1, 2, 3, 4, or 5:")
    if cluster_number and cluster_number.isdigit() and int(cluster_number) in {1, 2, 3, 4, 5}:
        cluster_number = int(cluster_number)
        break
    else:
        messagebox.showerror("Error", "Invalid Cluster ID. Please enter a valid number (1, 2, 3, 4, or 5).")


# # Run the models with default and optimal value

# In[7]:


# Subset training dataset keeping only the correct cluster
current_df = df[df['Cluster'] == cluster_number - 1 ]

# Define the features (X) and target (y)
X = current_df.drop(columns=['upt']) 
X = X.drop(columns=['Cluster'])
y = current_df['upt']


# In[10]:


# Read optimal parameters
file_path = os.path.join(current_directory, 'model_optimal_parameters.csv')
optimal_parameters = pd.read_csv(file_path)

# Subset the parameters only to the correct cluster
optimal_row = optimal_parameters[(optimal_parameters['cluster'] == str(cluster_number - 1)) & (optimal_parameters['database'] == "df")].iloc[0]

# Define all the parameters
random_state_value = 42
rf_n_estimators_value = int(optimal_row['rf_n_estimators'])
rf_max_depth_value = optimal_row['rf_max_depth']
rf_max_depth_value = None if rf_max_depth_value == 'None' else int(rf_max_depth_value)
rf_min_samples_split_value = int(optimal_row['rf_min_samples_split'])
rf_criterion_value = optimal_row['rf_criterion']
rf_min_samples_leaf_value = int(optimal_row['rf_min_samples_leaf'])
rf_max_features_value = None if pd.isna(optimal_row['rf_max_features']) else optimal_row['rf_max_features']
svr_C_value = optimal_row['svr_C']
svr_epsilon_value = optimal_row['svr_epsilon']
gbr_n_estimators_value = int(optimal_row['gbr_n_estimators'])
gbr_max_depth_value = optimal_row['gbr_max_depth']
gbr_max_depth_value = None if gbr_max_depth_value == 'None' else int(gbr_max_depth_value)
gbr_learning_rate_value = optimal_row['gbr_learning_rate']
gbr_min_samples_split_value = int(optimal_row['gbr_min_samples_split'])
gbr_min_samples_leaf_value = int(optimal_row['gbr_min_samples_leaf'])
gbr_max_features_value = None if pd.isna(optimal_row['gbr_max_features']) else optimal_row['gbr_max_features']
gbr_subsample_value = optimal_row['subsample']
gbr_criterion_value = optimal_row['grb_criterion']


# In[11]:


# Define models
models = {
    'rf': RandomForestRegressor(
        n_estimators=rf_n_estimators_value,
        n_jobs=-1,
        max_depth=rf_max_depth_value,
        min_samples_split=rf_min_samples_split_value,
        random_state=random_state_value,
        criterion=rf_criterion_value,
        min_samples_leaf=rf_min_samples_leaf_value,
        max_features=rf_max_features_value
    ),
    'svr': SVR(
        kernel='linear',
        C=svr_C_value,
        epsilon=svr_epsilon_value
    ),
    'gbr': GradientBoostingRegressor(
        n_estimators=gbr_n_estimators_value,
        criterion=gbr_criterion_value,
        learning_rate=gbr_learning_rate_value,
        min_samples_split=gbr_min_samples_split_value,
        max_depth=gbr_max_depth_value,
        random_state=random_state_value,
        min_samples_leaf=gbr_min_samples_leaf_value,
        max_features=gbr_max_features_value,
        subsample=gbr_subsample_value
    )
}

# Train each model
trained_models = {}
for model_name, model in models.items():
    model.fit(X, y)
    trained_models[model_name] = model


# # Predict data

# In[12]:


# Initialize vectors to store predictions
rf_predictions = []
svr_predictions = []
gbr_predictions = []

# Predict data for each model and store predictions in vectors
for model_name, model in trained_models.items():
    # Make predictions
    predictions = model.predict(x_prediction)
    
    # Store predictions in the corresponding vector
    if model_name == 'rf':
        rf_predictions = predictions
    elif model_name == 'svr':
        svr_predictions = predictions
    elif model_name == 'gbr':
        gbr_predictions = predictions

# Add predictions to the DataFrame as new columns
x_prediction['rf_prediction'] = rf_predictions
x_prediction['svr_prediction'] = svr_predictions
x_prediction['gbr_prediction'] = gbr_predictions


# In[14]:


# Optionally, save the DataFrame with predictions to a new CSV file
output_file_path = os.path.join(current_directory, 'predictions_results.csv')
x_prediction.to_csv(output_file_path, index=False)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Filter the feature importance DataFrame
feature_importance_filtered = feature_importance[(feature_importance['df'] == 'df') & 
                                                 (feature_importance['run_type'] == 'optimal')]

# Drop the 'df' and 'run_type' columns
feature_importance_filtered = feature_importance_filtered.drop(columns=['df', 'run_type'])

# Keep only the top 15 highest importance values per 'model' group
top_features  = feature_importance_filtered.groupby(['model', 'cluster']).apply(lambda x: x.nlargest(10, 'importance')).reset_index(drop=True)

# Classify the type of variable
selected_features = ['OTP', 'fare', 'speed', 'voms', 'vrh','frequency_peak']
top_features['Variables'] = np.where(top_features['feature'].isin(selected_features), 'Operational', 'Sociodemographic')

top_features['model'] = top_features['model'].replace({'rf': 'Random Forest', 'gbr': 'Gradient Boosting'})

top_features['importance'] = top_features['importance'].mask(top_features['importance'] < 0.001, np.nan)


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Round up importance to the closest 0.2
max_importance = np.ceil(top_features['importance'].max() * 10) / 10
top_features['importance_cat'] = np.ceil(top_features['importance'] * 5) / 5

# Define the colors for the custom colormap
colors = sns.color_palette("viridis", n_colors=9).as_hex()

# Create a dictionary mapping importance to color
color_map = {imp: color for imp, color in zip(np.arange(0, max_importance + 0.1, 0.1), colors)}

# Create a custom colormap with discrete colors
cmap = mcolors.ListedColormap([color_map[imp] for imp in np.arange(0, max_importance + 0.1, 0.1)])

# Define a function to plot heatmap
def plot_heatmap(data, **kwargs):
    data = data.pivot(index='feature', columns='cluster', values='importance_cat')
    ax = sns.heatmap(data, cmap=cmap, linewidths=1, linecolor='black', cbar=False, **kwargs)
    return ax

# Create FacetGrid with adjusted height and aspect ratio
g = sns.FacetGrid(top_features, col='model', row='Variables', height=8, aspect=0.5, 
                   row_order=['Operational', 'Sociodemographic'], 
                   gridspec_kws={'height_ratios': [6, 25], 'hspace': 0.05},  # Specify height ratios for rows
                   sharey='row',
                   margin_titles=True)  # Place titles at the top and left

# Add the heatmap
g.map_dataframe(plot_heatmap)
g.set_axis_labels("Clusters", "Variables")

# Add color bar and adjust layout
cbar_ax = g.fig.add_axes([0.95, 0.25, 0.03, 0.5])  # Add a colorbar axis on the right side of the grid
bounds = np.arange(0, max_importance + 0.1, 0.1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
g.fig.colorbar(sm, ticks=bounds, cax=cbar_ax, orientation='vertical')  # Set orientation to vertical

# Adjust the title and layout
plt.show()
plt.savefig("feature_importance.png")


# In[ ]:


top_features['feature'].unique().tolist()

