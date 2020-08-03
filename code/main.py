import numpy as np
from generate_initial_data import generate_initial_data
from clean_up_data import clearn_up_data
from generating_k_fold import generating_k_fold
from tunning_hyper_parameter_random_forest_regressor import tunning_hyper_parameter_random_forest_regressor
from finding_worst_prediction_and_return_suggested_inqury_points import finding_worst_prediction_and_return_suggested_inqury_points
from generate_data_for_active_learning import active_learning_query_point
from plot_val_vs_ground_truth import plot_val_vs_ground_truth

if __name__ == '__main__':

    #some program arguments
    k_fold = 5 #number of fold for cross validation
    max_depth = 12 #depth of the tree in random forest
    max_num_of_trees = 1000 #number of trees in random forest
    num_of_correction_points = 10 #number of points where we want to sample more
    num_sample_for_each_bad_point = 10 #number of samples for each correction point
    max_iteration = 10 #how many times we want to run active learning
    iteration_counter=0

    #parameters for accelerator
    #the following threshold_cut only works for the dcgun example
    threshold_cut={}
    threshold_cut['qb']=[0,5000]
    threshold_cut['max_enxy']=[0,30]


    #the first step of the program is to generate initial training data
    data_name_list = generate_initial_data(num_sample_for_each_bad_point*num_of_correction_points*2)

    while iteration_counter<max_iteration:
        #the second step of the program is to clean up the outlier of the generated data
        [total_data_x, total_data_y, features_list, targets_list] = \
            clearn_up_data(data_name_list[0], data_name_list[1])

        if iteration_counter>=1:
            #here, I need to concatenate new and old data
            total_data_x = np.concatenate((old_x_data, total_data_x), axis=0)
            total_data_y = np.concatenate((old_y_data, total_data_y), axis=0)

        #the third step of the program is to turn the cleaned data into k-folds
        fold_list = generating_k_fold(k_fold, total_data_x, total_data_y)

        #the fourth step of the program is to learn the hyperparameter
        best_model_parameter = tunning_hyper_parameter_random_forest_regressor(max_depth, max_num_of_trees, fold_list, total_data_x, total_data_y)

        plot_val_vs_ground_truth(best_model_parameter, fold_list, total_data_x, total_data_y, targets_list, iteration_counter, threshold_cut)

        if iteration_counter<max_iteration-1:
            #the fifth step of the program is to generate new inqury points
            [suggest_output_bound, suggest_input_bound] = finding_worst_prediction_and_return_suggested_inqury_points \
                (best_model_parameter, fold_list, total_data_x, total_data_y, features_list, targets_list, num_of_correction_points)

            #the sixth step of the program is to run GPT again based on new_inqury_points
            data_name_list = active_learning_query_point(suggest_output_bound, suggest_input_bound, threshold_cut, \
                                                         num_sample_for_each_bad_point, num_of_correction_points, iteration_counter)

            old_x_data = total_data_x
            old_y_data = total_data_y

        iteration_counter = iteration_counter+1