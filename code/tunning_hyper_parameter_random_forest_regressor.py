from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def tunning_hyper_parameter_random_forest_regressor(max_depth,max_num_of_trees,fold_list,total_data_x,total_data_y):

    #first, I need to know what hyperparameters we are tunning
    p1_max = max_depth
    p1_interval = 2
    p2_max = max_num_of_trees
    p2_interval = 100
    parameter_1_list = []
    parameter_2_list = []
    for index1 in range(2, p1_max, p1_interval):
        parameter_1_list.append(index1)
    for index2 in range(100, p2_max, p2_interval):
        parameter_2_list.append(index2)

    plot_mse = np.zeros((len(parameter_1_list), len(parameter_2_list)))
    #i am doing mes because I am doing regression




    #second, I am going to tablulate the mse loss
    print('begin to tune hyperparameter')
    plot_x=[]
    plot_y=[]
    plot_z=[]
    counter = 0 #this counter is just helping me to compute percentage

    for index1 in range(0, len(parameter_1_list)):
        for index2 in range(0, len(parameter_2_list)):
            p1 = parameter_1_list[index1]
            p2 = parameter_2_list[index2]
            regr = RandomForestRegressor(max_depth=p1, random_state=0,
                                         n_estimators=p2)
            total_mse_error = 0
            for index in range(0, len(fold_list)):
                X = total_data_x[fold_list[index].train_index]
                Y = total_data_y[fold_list[index].train_index]
                regr.fit(X, Y)
                val_x = total_data_x[fold_list[index].val_index]
                val_y = total_data_y[fold_list[index].val_index]
                prediction = regr.predict(val_x)

                # now, I compute the mse error
                mse_error = prediction - val_y
                #at this point, just after subtraction, each row of mse_error is one data point
                mse_error = sum(sum(np.transpose(mse_error * mse_error))) / prediction.shape[0]
                #for a matrix, the sum command first sum column by column
                #the multiplication sign means just elementwise multiplication
                #the sum(transpose()) part means to sum up the error contribution from each dimension
                #the outermost sum means summing up the entire error

                total_mse_error = total_mse_error + mse_error


            total_mse_error = total_mse_error / len(fold_list) * 1.0
            # total_mse_error
            plot_mse[index1, index2] = total_mse_error
            plot_x.append(p1)
            plot_y.append(p2)
            plot_z.append(total_mse_error)
            counter = counter + 1
            print('finished percentage', counter / (len(parameter_1_list) * len(parameter_2_list)))


    #before going further, we want to visualize the process of tunning hyperparameter
    ax = plt.axes(projection='3d')
    x=plot_x
    y=plot_y
    z=plot_z
    ax.scatter(x,y,z,c='b',marker='o')
    ax.set_xlabel('tree depth')
    ax.set_ylabel('forest size')
    ax.set_zlabel('mse_error')
    error_std = np.std(plot_z)
    ax.set_zlim(min(plot_z)-0.2*error_std,min(plot_z)+error_std)
    plt.savefig('hyper_parameter_plot')




    #the third step is to select the best hyperparameter
    best_p1 = 0
    best_p2 = 0
    loss = 1e9
    for row in range(0, plot_mse.shape[0]):
        for column in range(0, plot_mse.shape[1]):
            if plot_mse[row, column] < loss:
                loss = plot_mse[row, column]
                best_p1 = parameter_1_list[row]
                best_p2 = parameter_2_list[column]


    print('DONE WITH TUNNING HYPERPARAMETER')
    print('tree depth should be',best_p1)
    print('forest size should be',best_p2)

    return [best_p1,best_p2]