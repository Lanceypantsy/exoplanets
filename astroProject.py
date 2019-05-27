##### Machine Learning Mid-Term Project
##### Lance Barto
##### Dr. Feng Jiang


# Initial Imports
import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt

##### Import the training and testing data ############
##### Also, partition the data into four groups #######
##### for the sake of being able to simply create #####
##### exploratory visualizations, etc.... #############

# Import the training and testing data 
data_labelled = pd.read_csv('exoTrain.csv')
testing_data_labelled = pd.read_csv('exoTest.csv')

# Save the testing labels before dropping the columns
training_labels = data_labelled.LABEL
testing_labels = testing_data_labelled.LABEL

# Drop the label columns now that they're saved for 
# later testing/training
data = data_labelled.drop(columns='LABEL')
testing_data = testing_data_labelled.drop(columns='LABEL')

training_exo = data[training_labels==2]
training_non = data[training_labels==1]

# Under-sample the training set
training_non = training_non.sample(2000, random_state=999)

# 
testing_exo = testing_data[testing_labels==2]
testing_non = testing_data[testing_labels==1]

#####################################
###### Visualization Functions ######
#####################################

# Create a visualization function to regularly view the 
# results of the data cleaning/massage for the positive
# exoplanet examples.

def plot_exo(df, num_samp, col_range):
    
    # Define the plot object, and size. We are setting a high dpi
    # parameter, knowing that these plots are destined for print.
    fig = plt.figure(figsize=(4, 4), dpi=300)

    # A np array to represent each of our flux samples over the 37 
    # positive samples. 
    x = np.array(range(col_range))

    for i in np.arange(num_samp):
        ax = fig.add_subplot(num_samp // 3 + 1, 3, i+1,)
        ax.scatter(x, df.iloc[i, :], marker='.', color='black', alpha=.2)
        ax.set_title('Exoplanet Star {}'.format(i+1))
        ax.set_xticks([])


# Create a function for plotting the non-exoplanets, randomly
# sampling the appropriate number to fit the grid.

def plot_non(df, num_samp, col_range, r_state):
    
    # Create a list of 39 samples (may as well fill our grid), all 
    # selected from row 37 or later, knowing that those are our
    # negative samples.
    negative_samples = df.sample(num_samp, random_state=r_state)
    
    fig = plt.figure(figsize=(4, 4), dpi=300)

    # A np array to represent each of our flux samples over the 37 
    # positive samples. 
    x = np.array(range(col_range))

    for i in np.arange(num_samp):
        ax = fig.add_subplot(num_samp // 3 + 1, 3, i+1)
        ax.scatter(x, df.iloc[i,:], marker='.', color='black', alpha=.2)
        ax.set_title('Non-exoplanet Star {}'.format(i+1))
        ax.set_xticks([])

# Create a Function to look at portions of a given exoplanet sample, for 
# the sake of looking at samples in a more clear, larger window,
# next to the entire time-series

def plot_zoom(df, beg_index, end_index, star_num):

    # Define the figure size, and dpi
    fig= plt.figure(figsize=(6,6), dpi=300)

    # Zoomed in to desired flux samples
    ax = fig.add_subplot(1, 2, 1)
    plt.scatter(np.arange(end_index-beg_index),
                df.iloc[star_num-1, beg_index:end_index], 
                marker='.', color='black', alpha=.2)
    
    plt.title(str(end_index-beg_index) + 
              ' flux samples from Exoplanet Star #' + str(star_num) +
              '\n Flux.' + str(beg_index+1) + ' to ' + 'Flux.' +
              str(end_index))
    
    plt.xticks([])

    ax = fig.add_subplot(1, 2, 2)
    plt.scatter(np.arange(len(df.iloc[0, :])), df.iloc[star_num-1, :], marker='.',
                color='black', alpha=.2)
    plt.title('All flux samples Exoplanet Star #' + str(star_num))
    plt.xticks([])

    plt.show()

# Create a Function to look at portions of a given non-exoplanet sample, for 
# the sake of looking at samples in a more clear, larger window,
# next to the entire time-series

def plot_zoom_non(df, beg_index, end_index, star_num):
    # Define the figure size, and dpi
    fig= plt.figure(figsize=(6,6), dpi=300)

    # Zoomed in to desired flux samples
    ax = fig.add_subplot(1, 2, 1)
    plt.scatter(np.arange(end_index-beg_index),
                df.iloc[star_num-1, beg_index:end_index], 
                marker='.', color='black', alpha=.2)
    
    plt.title(str(end_index-beg_index) + 
              ' flux samples from Non-exoplanet Star #' + str(star_num) +
              '\n Flux.' + str(beg_index+1) + ' to ' + 'Flux.' +
              str(end_index))
    
    plt.xticks([])

    ax = fig.add_subplot(1, 2, 2)
    plt.scatter(np.arange(len(df.iloc[0, :])), df.iloc[star_num-1, :], marker='.',
                color='black', alpha=.2)
    plt.title('All flux samples Non-exoplanet Star #' + str(star_num))
    plt.xticks([])

    plt.show()


##############################################
############ Exploratory visualizations ######
##############################################

# # The first round of plotting the un-processed samples
# plot_exo(training_exo, 37, 3197)
# plot_non(training_non, 37, 3197, 99)
#
# # A zoomed-in view of an exoplanet sample with a clear planet
# # traversal
# plot_zoom(training_exo, 0, 500, 12)

##############################################
########## Outlier Removal Functions #########
##############################################

# A function to remove upper outliers, and replace them with an
# average value determined by their neighbors. Half-width refers
# to number of neighbors on each side to calculate the replacement
# value


def reduce_upper(df, reduce=0.01, half_width=4):
    length = len(df.iloc[0, :])
    remove = int(length*reduce)
    
    for i in df.index.values:
            values = df.loc[i, :]
            sorted_values = values.sort_values(ascending=False)
            
            for j in range(remove):
                    idx = sorted_values.index[j]
                    
                    new_val = 0
                    count = 0
                    idx_num = (int(idx[5:]))
                    
                    for k in range(2 * half_width + 1):
                        idx2 = idx_num + k - half_width
                        
                        if idx2< 1 or idx2 >= length or idx_num == idx2:
                            continue
                            
                        new_val += values['FLUX.' + str(idx2)]
                        
                        count += 1
                    
                    new_val /= count
                    
                    if new_val < values[idx]:
                        df.set_value(i, idx, new_val)
                        
    return df

# A function to remove lower outliers, and replace them with an
# average value determined by their neighbors. Half-width refers
# to number of neighbors on each side to calculate the replacement
# value


def reduce_lower(df, reduce=0.01, half_width=4):
    length = len(df.iloc[0, :])
    remove = int(length*reduce)

    for i in df.index.values:
            values = df.loc[i, :]
            sorted_values = values.sort_values(ascending=True)
            
            for j in range(remove):
                    idx = sorted_values.index[j]
                    
                    new_val = 0
                    count = 0
                    idx_num = (int(idx[5:]))
                    
                    for k in range(2 * half_width + 1):
                        idx2 = idx_num + k - half_width
                        
                        if idx2< 1 or idx2 >= length or idx_num == idx2:
                            continue
                            
                        new_val += values['FLUX.' + str(idx2)]
                        
                        count += 1
                    
                    new_val /= count
                    
                    if new_val < values[idx]:
                        df.set_value(i, idx, new_val)
                        
    return df

###############################################
####### Apply Outlier Removal #################
###############################################

# Apply the upper outlier removal function twice, each time
# replacing the highest 1% of values with the means of their
# neighbors

for i in np.arange(2):
    training_exo = reduce_upper(training_exo)
    training_non = reduce_upper(training_non)
    testing_exo = reduce_upper(testing_exo)
    testing_non = reduce_upper(testing_non)

# Imports
from scipy.signal import savgol_filter
from scipy.signal import medfilt

# A function which applys each of these two filters,
# a median smoothing filter, and a savitsky-golay
# smoothing filter.

#################################################
############## Smoothing Function ###############
#################################################

# This function uses both median and sovitsky-golay 
# smoothing filters. The output returned is a list of np
# arrays.


def short_transit_filter(df):
    
    length = df.shape[0]
    output = []
    
    for i in range(length):
        
        y0 = medfilt(df.iloc[i,:], 41)
        
        for idx in range(len(y0)):
            y0[idx] = df.iloc[i, idx] - y0[idx]
        
        y1 = savgol_filter(y0, 21, 3, deriv=0)
        output.append(y1)
    
    return output

##################################################
################ Apply Smoothing #################
##################################################

# Apply the smoothing to all of the partitioned dataframes

test1_training_exo = short_transit_filter(training_exo)
test1_training_non = short_transit_filter(training_non)
test1_testing_exo = short_transit_filter(testing_exo)
test1_testing_non = short_transit_filter(testing_non)

##################################################
########## Re_DataFrame Function #################
##################################################

# Make a list of column names to apply
col_names = []

for i in np.arange(3197):
    col_names.append('FLUX.' + str(i+1))

# A simple function to recast the data into a dataframe
# and rename the columns so we can use the outlier removal
# tools again


def re_dataframe(list_of_arrays, col_names):
    
    df = pd.DataFrame(list_of_arrays)
    df.columns = col_names
    
    return(df)

##################################################
############# Return the data from ###############
######### smoothing functions to dataframes ######
##################################################


test2_training_exo = re_dataframe(test1_training_exo, col_names)
test2_training_non = re_dataframe(test1_training_non, col_names)
test2_testing_exo = re_dataframe(test1_testing_exo, col_names)
test2_testing_non = re_dataframe(test1_testing_non, col_names)

##################################################
####### Visualize the data after the processing ##
##################################################

# plot_exo(test2_training_exo, 37, 3197)
# plot_non(test2_training_non, 39, 3197, r_state=99)

##################################################
######## Further Processing ######################
##################################################

# This time, we will apply the upper outlier removal tool,
# with a much wider half-width, because we have some extreme
# outliers on the 'edges' of our time series.

test3_training_exo = reduce_upper(test2_training_exo,  
                                  reduce=.02, half_width=50)

test3_training_non = reduce_upper(test2_training_non,
                                  reduce=.02, half_width=50)

test3_testing_exo = reduce_upper(test2_testing_exo,
                                 reduce=.02, half_width=50)

test3_testing_non = reduce_upper(test2_testing_non,
                                 reduce=.02, half_width=50)

####################################################
############## A round of visualizations to ########
############ determine efficacy of results #########
####################################################

# plot_exo(test3_training_exo, 37, 3197)
# plot_non(test3_training_non, 39, 3197, r_state=99)

# Three exoplanet cases with extreme outliers which is making it
# difficult to identify features, let's zoom in on them.

# plot_zoom(test3_training_exo, 75, 3022, 31)
# plot_zoom(test3_training_exo, 75, 3022, 15)
# plot_zoom(test3_training_exo, 75, 3022, 26)

# # A closer look at a non-exoplanet sample after the processing
# plot_zoom_non(test3_training_non, 75, 3022, 21)

###################################################
######### Time-series trimming for outlier ########
################## removal ########################
###################################################

# Trim the time-series to eliminate outliers at the 'edges'
# which were created during processing.
test4_training_exo = test3_training_exo.iloc[:, 74:3022]
test4_training_non = test3_training_non.iloc[:, 74:3022]
test4_testing_exo = test3_testing_exo.iloc[:, 74:3022]
test4_testing_non = test3_testing_non.iloc[:, 74:3022]

###################################################
############ Another Round of smoothing ###########
###################################################

# Run the smoothing function 
test5_training_exo = short_transit_filter(test4_training_exo)
test5_training_non = short_transit_filter(test4_training_non)
test5_testing_exo = short_transit_filter(test4_testing_exo)
test5_testing_non = short_transit_filter(test4_testing_non)

# No need to re-name columns, as we are no longer going
# to use the upper and lower outlier removal tools
test6_training_exo = pd.DataFrame(test5_training_exo)
test6_training_non = pd.DataFrame(test5_training_non)
test6_testing_exo = pd.DataFrame(test5_testing_exo)
test6_testing_non = pd.DataFrame(test5_testing_non)

##################################################
############ Final processed visualizations ######
##################################################

# exoplanets
# plot_exo(test6_training_exo, 37, 2948)

# non-exoplanets
# plot_non(test6_training_non, 39, 2948, r_state=99)

##################################################
########### Data Synthesizations #################
##################################################

# Read the samples in reverse, and append to the original
# samples, doubling the training set size. 
test7 = test6_training_exo[test6_training_exo.columns[::-1]]

test7.columns = np.arange(2948)

test7_training_exo = test6_training_exo.append(test7)


# Concatenate the sample with themselves, and then drop every second 
# column to return to the original size
df3 = pd.concat([test7_training_exo, test7_training_exo], axis=1)
df4 = df3.iloc[:, ::2]

df4.columns = np.arange(2948)

# Concatenate the sample with themselves twice, and then drop every 2/3 
# columns to return to the original size
df5 = pd.concat([test7_training_exo, test7_training_exo, test7_training_exo], axis=1)
df6 = df5.iloc[:, ::3]

df6.columns = np.arange(2948)

# Concatenate the sample with themselves 3 times, and then drop every 3/4 
# columns to return to the original size
df7 = pd.concat([test7_training_exo, test7_training_exo, test7_training_exo, test7_training_exo], axis=1)
df8 = df7.iloc[:, ::4]

df8.columns = np.arange(2948)

# Aggregate the new/synthesized data back into one training set
test8_training_exo = test7_training_exo.append(df4)
test8_training_exo = test8_training_exo.append(df6)
test8_training_exo = test8_training_exo.append(df8)

##################################################
########### Visualize the syntethesized samples ##
##################################################

# plot_exo(test8_training_exo, 198, 2948)

##################################################
######### Clean up labels, aggregate data ########
######### to prepare for SMOTE ###################
##################################################

# Create labels the appropriate length
training_labels_exo = np.repeat(1, 296)
training_labels_non = np.repeat(0, 2000)

# Concatenate the labels appropriately
train_Y = np.concatenate([training_labels_exo, training_labels_non])

# Create labels the appropriate length
testing_labels_exo = np.repeat(1, 5)
testing_labels_non = np.repeat(0, 565)

# Concatenate the labels appropriately
test_Y = np.concatenate([test6_testing_exo, test6_testing_non])

# Recombine the data into a single training set
train_X = test8_training_exo.append(test6_training_non)

# Recombine the data into a single testing set
test_X = test6_testing_exo.append(test6_testing_non)

# Export the data to np arrays to apply SMOTE and 
# for model building
train_X = train_X.values
test_X = test_X.values

################################################
############# Apply SMOTE ######################
################################################

# Import SMOTE Tool
from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(train_Y==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(train_Y==0)))

# Over-sample the minority class for a 40/60 final class balance
sm = SMOTE(.8)
train_X, train_Y = sm.fit_sample(train_X, train_Y.ravel())

print('After OverSampling, the shape of train_X: {}'.format(train_X.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(train_Y.shape))

print("After OverSampling, counts of label '1': {}".format(sum(train_Y==1)))
print("After OverSampling, counts of label '0': {}".format(sum(train_Y==0)))

##################################################
####### Visualize a sample of the SMOTE ##########
############ created data points #################
##################################################

# Create Sample
synthetic_samples = pd.DataFrame(train_X[2296:])

# # Visualize 39 SMOTE samples
# plot_exo(synthetic_samples, 39, 2948)
