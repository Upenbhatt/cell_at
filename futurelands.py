# -*- coding: utf-8 -*-
"""
Created on Oct 2023

@author: Angelos Alamanos
"""
#####  Data Preprocessing - Estimate the Transition Probability Matrices
# Using the Change Matrices (tabulate areas) from ArcGIS (replace with your own values)

import numpy as np

# Example change matrix for a specific year change (e.g., from 2017 to 2019)
change_matrix_17to19 = np.array([
    [1731058, 133196, 315692, 804680, 1723058],   # From Water (1)
    [5299, 240296532, 855079, 2544138, 2821032],  # From Urban (2)
    [128296, 5769561, 43382158, 18380158, 9868863],   # From Barren Land (3)
    [271093, 6471544, 17563778, 332291024, 6269949],     # From Forest (4)
    [169595, 7868211, 1984952, 3105225, 163983764]   # From Crops (5)
])

change_matrix_19to21 = np.array([
    [777481, 38699, 250693, 626384, 612085],   # From Water (1)
    [8699, 252244745, 1651860, 3080726, 3553014],  # From Urban (2)
    [33799, 2330644, 38908766, 17874071, 4954381],   # From Barren Land (3)
    [184595, 7561318, 15326632, 328873606, 5179075],     # From Forest (4)
    [207895, 9647968, 3383518, 4464892, 166962392]   # From Crops (5)
])

change_matrix_21to23 = np.array([
    [495800, 12500, 393900, 193900, 116400],   # From Water (1)
    [6700, 265009800, 1221700, 2891700, 2700000],  # From Urban (2)
    [83800, 3108100, 34895500, 18330800, 3104700],   # From Barren Land (3)
    [231200, 5989900, 16121700, 328010100, 4575300],     # From Forest (4)
    [219000, 11574900, 3490900, 3950900, 162029600]   # From Crops (5)
])

# Normalize the Change Matrices
total_transitions1 = np.sum(change_matrix_17to19, axis=1)
normalized_matrix1 = change_matrix_17to19 / total_transitions1[:, np.newaxis]

total_transitions2 = np.sum(change_matrix_19to21, axis=1)
normalized_matrix2 = change_matrix_19to21 / total_transitions2[:, np.newaxis]

total_transitions3 = np.sum(change_matrix_21to23, axis=1)
normalized_matrix3 = change_matrix_21to23 / total_transitions3[:, np.newaxis]


# Step 4: Construct the Transition Probability Matrix
transition_probability_matrix1 = normalized_matrix1
transition_probability_matrix2 = normalized_matrix2
transition_probability_matrix3 = normalized_matrix3

np.set_printoptions(precision=3, suppress=True)

# Display the resulting transition probability matrix
np.set_printoptions(precision=4, suppress=True)
print(transition_probability_matrix1)
print(transition_probability_matrix2)
print(transition_probability_matrix3)



##################### 2016 to 2021 #############################

import numpy as np
import os
import rasterio
from rasterio.transform import from_origin

# Define the data directory and file paths
data_directory = r'D:\SLIS-G\Cellular_automata\New_method\reclassified_lulc'
land_use_2017 = r'D:\SLIS-G\Cellular_automata\New_method\reclassified_lulc\2017_lulc1.tif'

# Define the transition probabilities matrix manually
transition_probs = np.array([
    [0.998, 0.001, 0.001, 0.000, 0.000],
    [0.000, 1.000, 0.000, 0.000, 0.000],
    [0.009, 0.020, 0.928, 0.043, 0.000],
    [0.000, 0.002, 0.001, 0.995, 0.002],
    [0.000, 0.002, 0.002, 0.000, 0.996]
   ])

# Check if the probabilities are valid (between 0 and 1 and sum to 1 for each row)
if (transition_probs < 0).any() or (transition_probs > 1).any() or not np.allclose(transition_probs.sum(axis=1), 1):
    raise ValueError("Invalid transition probabilities.") 

# Load the 2016 land use map using rasterio
with rasterio.open(land_use_2017) as src:
    current_land_use_map_2017 = src.read(1)
    transform = src.transform  # Get the spatial transform from the input raster

# Define a function to apply the transition based on probabilities
def apply_transition(land_use, transition_probs):
    new_land_use = np.copy(land_use)
    rows, cols = land_use.shape
    for row in range(rows):
        for col in range(cols):
            current_category = int(land_use[row, col])
            if 1 <= current_category <= 5:
                transition_probs_normalized = transition_probs[current_category - 1]
                new_category = np.argmax(np.random.multinomial(1, transition_probs_normalized))
                new_land_use[row, col] = new_category + 1
    return new_land_use

# Apply the transition to the 2022 land use map
predicted_land_use_2022 = apply_transition(current_land_use_map_2017, transition_probs)

# Get the shape of the predicted_land_use_2022 array
rows, cols = predicted_land_use_2022.shape

# Save the predicted land use map for 2021 with spatial reference information
output_raster_path = os.path.join(data_directory, 'predicted_land_use_2022.tif')
with rasterio.open(output_raster_path, 'w', driver='GTiff', width=cols, height=rows, count=1, dtype=rasterio.int32, crs=src.crs, transform=transform) as dst:
    dst.write(predicted_land_use_2022, 1)

print("Prediction completed. The result is saved to:", output_raster_path)


###################  Repeat the same script for all time-steps needed  ##################
###########################  e.g.   2021 to 2026,  etc.  ################################
# It is recommended that you paste the above script as many times as necessary
# rather than always editing the one above, having thus a 'secure' template for the process.



###########################  VALIDATION   ##########################################

import geopandas as gpd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
import numpy as np

# Paths to truth and predicted point feature classes
truth_path = r'D:\SLIS-G\Cellular_automata\New_method\predicted\validactual21.shp'
predicted_path = r'D:\SLIS-G\Cellular_automata\New_method\predicted\validpred21.shp'


# Load the truth and predicted point feature classes
truth_data = gpd.read_file(truth_path)
predicted_data = gpd.read_file(predicted_path)

# Round spatial coordinates to 6 decimal places to ensure correct merging
truth_data['rounded_x'] = truth_data.geometry.x.round(6)
truth_data['rounded_y'] = truth_data.geometry.y.round(6)
predicted_data['rounded_x'] = predicted_data.geometry.x.round(6)
predicted_data['rounded_y'] = predicted_data.geometry.y.round(6)

# Generate a unique identifier based on spatial coordinates to ensure that the same points are compared
truth_data['unique_id'] = truth_data.geometry.apply(lambda geom: f'{geom.x:.6f}_{geom.y:.6f}')
predicted_data['unique_id'] = predicted_data.geometry.apply(lambda geom: f'{geom.x:.6f}_{geom.y:.6f}')

# Merge the datasets based on the 'unique_id' column
merged_data = truth_data.merge(predicted_data, on='unique_id', how='inner') #merge on geometry or on unique_id or on pointid

# Check if the datasets have the same number of points
if len(truth_data) != len(predicted_data):
    print("Warning: The number of points in truth and predicted datasets is not the same.")
    
# Check if there are common points
if len(merged_data) == 0:
    print("Error: There are no common points between the truth and predicted datasets.")
else:
    # Calculate accuracy metrics
    truth_labels = merged_data['grid_code_x'].astype(int)
    predicted_labels = merged_data['grid_code_y'].astype(int)
    
    
# Check the datasets and their column names in the data 
print(truth_data)
print(predicted_data)
print(merged_data)
# To ensure that we are comparing the correct ones and the merged file looks as expected
print(truth_data.columns)
print(predicted_data.columns)
print(merged_data.columns)


# Plot the truth_data and predicted_data points together, to ensure they are the same points
import matplotlib.pyplot as plt
ax = truth_data.plot(color='blue', label='Truth Data')
predicted_data.plot(ax=ax, color='red', label='Predicted Data')
plt.legend()
plt.show()

# Accuracy metrics
accuracy = accuracy_score(truth_labels, predicted_labels)
mae = mean_absolute_error(truth_labels, predicted_labels)
rmse = mean_squared_error(truth_labels, predicted_labels, squared=False)  # Use squared=False for RMSE
kappa = cohen_kappa_score(truth_labels, predicted_labels)
confusion = confusion_matrix(truth_labels, predicted_labels)

# Calculate precision, recall, and F1-score
classification_rep = classification_report(truth_labels, predicted_labels)

# Print the accuracy metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("Cohen's Kappa:", kappa)

print("Confusion Matrix:")
print(confusion)

# Print precision, recall, and F1-score
print("Classification Report:")
print(classification_rep)




######################  PLOT THE MAPS   ################################

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd

# Paths to the shapefiles for each year
shapefile_paths = [
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_shp\predicted25polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_shp\predicted27polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_shp\predicted29polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_shp\predicted31polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_shp\predicted33polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_shp\predicted35polygons.shp',
]

# Corresponding years
years = [2025, 2027, 2029, 2031, 2033, 2035]

# Land use categories and their colors
land_use_colors = {
    0: 'white',
    1: 'blue',
    2: 'gray',
    3: 'black',
    4: 'darkred',
    5: 'lightgreen',
}


# Land use labels for the legend
land_use_labels = {
    0: 'No data',
    1: 'Water',
    2: 'Urban',
    3: 'Barren Land',
    4: 'Forest',
    5: 'Crops',
}

# Create a figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loop through shapefiles and corresponding years
for i, (shapefile_path, year) in enumerate(zip(shapefile_paths, years)):
    # Read the shapefile using GeoPandas
    gdf = gpd.read_file(shapefile_path)

    # Get the axis for the current subplot
    ax = axes[i // 3, i % 3]

    # Plot the GeoDataFrame with specified colors and legend
    for land_use_code, color in land_use_colors.items():
        gdf[gdf['gridcode'] == land_use_code].plot(ax=ax, color=color)

    # Set title and remove axes
    ax.set_title(f'Year {year}', fontsize=16, fontweight='bold')
    ax.axis('off')

# Add a single legend for the 2035 map
ax_legend = axes[1, 2].inset_axes([0.85, 0.05, 0.35, 0.25])
ax_legend.axis('off')
for land_use_code, color in land_use_colors.items():
    land_use_label = {
        1: 'Water',
        2: 'Urban',
        3: 'Barren Land',
        4: 'Forest',
        5: 'Crops',
    }.get(land_use_code, f'Land Use {land_use_code}')
    ax_legend.add_patch(plt.Rectangle((0, (land_use_code - 1) * 0.2), 0.2, 0.2, color=color))
    ax_legend.annotate(land_use_label, (0.25, (land_use_code - 1) * 0.2), fontsize=12)

# Adjust subplot layout and remove gaps between subplots
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

# Show the figure
plt.show()




#########################  PLOT THE PREDICTED AREAS  ############################

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

# Paths to the shapefiles for each year
shapefile_paths = [
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_area_shp\predicted25polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_area_shp\predicted27polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_area_shp\predicted29polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_area_shp\predicted31polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_area_shp\predicted33polygons.shp',
    r'D:\SLIS-G\Cellular_automata\New_method\predicted_area_shp\predicted35polygons.shp',
    ]

# Corresponding years
years = [2025, 2027, 2029, 2031, 2033, 2035]

# Land use categories
land_use_categories = {
    0: 'No data',
    1: 'Water',
    2: 'Urban',
    3: 'Barren Land',
    4: 'Forest',
    5: 'Crops',
}

# Create an empty DataFrame to store land use category counts over time
land_use_counts = pd.DataFrame(columns=land_use_categories.values())

# Loop through shapefiles and corresponding years
for shapefile_path, year in zip(shapefile_paths, years):
    # Read the shapefile using GeoPandas
    gdf = gpd.read_file(shapefile_path)

    # Count the number of pixels for each land use category
    category_counts = {land_use_categories[i]: (gdf['gridcode'] == i).sum() for i in land_use_categories.keys()}

    # Append the counts to the DataFrame
    land_use_counts = land_use_counts.append(category_counts, ignore_index=True)

# Set up the plot
plt.figure(figsize=(10, 6))

# Define the bar colors
colors = [land_use_colors[col] for col in land_use_counts.columns]

# Create the bar plot
land_use_counts.plot(kind='bar', stacked=True, color=colors)

plt.title("Land Use Evolution Over Time")
plt.xlabel("Year")
plt.ylabel("Land Use Area (Number of Pixels)")

# Show the plot
plt.show()



