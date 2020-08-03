import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def plot_val_vs_ground_truth(best_model_parameter, fold_list, total_data_x, total_data_y, targets_list, iteration_counter, threshold_cut):

    #this is just for plotting
    #we want to visualize the data for understanding what the algorithm is doing
    
    best_p1 = best_model_parameter[0]
    best_p2 = best_model_parameter[1]
    regr = RandomForestRegressor(max_depth=best_p1, random_state=0,
                                 n_estimators=best_p2)
    fold_index = 0  #for plotting purpose, I am always using the 0th fold
    X = total_data_x[fold_list[fold_index].train_index]
    Y = total_data_y[fold_list[fold_index].train_index]
    regr.fit(X, Y)
    val_x = total_data_x[fold_list[fold_index].val_index]
    val_y = total_data_y[fold_list[fold_index].val_index]
    prediction = regr.predict(val_x)

    feature_index = 1 #for the dcgun example, the 1th feature is emittance

    sort_index = np.argsort(val_y[:, feature_index])
    samples = np.linspace(0, len(sort_index), len(sort_index))

    print('the shape of the sample number list is', samples.shape)
    print('the shape of the prediction is', prediction[sort_index, feature_index].shape)
    print('the type of sample number is', type(samples))
    print('the type of prediction is', type(prediction[sort_index, feature_index]))
    print('prediction is', prediction[sort_index, feature_index].tolist())
    print('sample number is', samples.tolist())

    temp_list = []
    for index in range(0,len(prediction[sort_index, feature_index].tolist())):
        temp_list.append(prediction[sort_index, feature_index].tolist()[index])

    x_list = []
    for index in range(0,len(samples.tolist())):
        x_list.append(samples.tolist()[index])

    plt.clf()
    for index in range(0,len(x_list)):
        plt.scatter(x_list[index], temp_list[index],c='r')
    plt.plot(samples.tolist(), val_y[sort_index, feature_index].tolist())
    plt.xlabel('sample number')
    plt.ylabel(targets_list[feature_index])
    plt.ylim(threshold_cut['max_enxy'])
    plot_name = 'ground_truth_vs_prediction_for_iteration_'+str(iteration_counter)
    plt.savefig(plot_name)
