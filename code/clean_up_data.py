import pickle
import numpy as np


def clearn_up_data(file_name_x,file_name_y):

    #here, I just load the data
    total_data_x = pickle.load(open(file_name_x, "rb"))
    total_data_y = pickle.load(open(file_name_y, "rb"))



    #here, I turn the structure of the data from pickle files into a numpy matrix
    #each rwo of the matrix is a data point
    #each column of the matrix is either a feature or a target, depending on whether it's the x matrix or the y matrix
    features_list = [feature for feature in total_data_x]
    targets_list = [feature for feature in total_data_y]
    Numsamples = len(total_data_x[features_list[0]])
    Numfeatures = len(total_data_x)
    Numtargets = len(total_data_y)

    x_data = np.zeros((Numsamples, Numfeatures))
    y_data = np.zeros((Numsamples, Numtargets))

    for j in range(len(features_list)):
        x_data[:, j] = total_data_x[features_list[j]]

    for j in range(len(targets_list)):
        y_data[:, j] = total_data_y[targets_list[j]]

    total_data_x = x_data
    total_data_y = y_data




    #now, I clean up the data based on one feature at a time
    for features_index in range(0,len(targets_list)):
        filtered_data = [] #the filtered_data keeps what is left, although this data is never used
        keep_index = []
        mean = np.mean(total_data_y[:, features_index])
        std = np.mean(total_data_y[:, features_index])

        for index in range(0, total_data_y.shape[0]):
            if total_data_y[index, features_index] >= mean - 2 * std and total_data_y[
                index, features_index] <= mean + 2 * std:
                filtered_data.append(total_data_y[index, features_index])
                keep_index.append(index)

        total_data_x = total_data_x[keep_index, :]
        total_data_y = total_data_y[keep_index, :]

    return [total_data_x,total_data_y,features_list,targets_list]