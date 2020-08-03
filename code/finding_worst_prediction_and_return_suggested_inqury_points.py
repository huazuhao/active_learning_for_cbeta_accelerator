from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

def finding_worst_prediction_and_return_suggested_inqury_points\
                (best_model_parameter,fold_list, total_x_data,total_y_data, features_list, targets_list, num_of_correction_points):

    #the first step is to create a random forest regressor based on the learned hyperparameter
    best_p1 = best_model_parameter[0]
    best_p2 = best_model_parameter[1]
    regr = RandomForestRegressor(max_depth=best_p1, random_state=0,
                                 n_estimators=best_p2)


    #the second step is to go through each fold and find the fold of data that contains the worst prediction
    print('begin to go through each fold and find the fold of data with the largest error')
    max_error_fold = 0
    max_error_fold_index = 0
    for index in range(0, len(fold_list)):
        X = total_x_data[fold_list[index].train_index]
        Y = total_y_data[fold_list[index].train_index]

        regr.fit(X, Y)

        val_x = total_x_data[fold_list[index].val_index]
        val_y = total_y_data[fold_list[index].val_index]

        prediction = regr.predict(val_x)

        feature_index = 1 #this is just a dummy variable used for sorting

        sort_index = np.argsort(val_y[:, feature_index])

        se_error = prediction[sort_index, :] - val_y[sort_index, :]
        se_error = se_error * se_error
        se_error = sum(np.transpose(se_error))

        r = sorted(se_error, reverse=True)
        r = sum(r[0:num_of_correction_points]) #for a set of validation data, get the first 5 largest error

        if r > max_error_fold:
            max_error_fold = r
            max_error_fold_index = index
        print('finished percentage', index / len(fold_list))
    print('the fold corresponds to the largest error is', max_error_fold_index)



    #now, given that we know which fold of data contains the largest error,
    #we will go into that folder of data and find the indexes that corresponds to the largest error
    index = max_error_fold_index
    X = total_x_data[fold_list[index].train_index]
    Y = total_y_data[fold_list[index].train_index]
    regr.fit(X, Y)
    val_x = total_x_data[fold_list[index].val_index]
    val_y = total_y_data[fold_list[index].val_index]


    prediction = regr.predict(val_x)
    feature_index = 1 #this is just a dummy variable used for sorting
    sort_index = np.argsort(val_y[:, feature_index])
    se_error = prediction[sort_index, :] - val_y[sort_index, :]
    se_error = se_error * se_error
    se_error = sum(np.transpose(se_error))
    large_error_index = se_error.argsort()[-num_of_correction_points:][::-1]



    #lastly, I return the new inqury points bound
    #here is the data structure
    #I retrun a list
    #a single element in the list a dictionary
    #each entry of the dictionary contains a feature and a value bound, with the keys of the dictionary being accelerator feature

    feature_and_range_dict = pickle.load( open("feature_and_range.p","rb"))
    suggest_input_bound=[]
    suggest_output_bound=[]
    for index in range(0, len(large_error_index)):
        temp = large_error_index[index]
        large_error_input = val_x[sort_index, :][temp]
        # large_error_entry now corresponds to the input value used in gpt for generating those large
        # prediction-ground-truth difference
        large_error_output = val_y[sort_index,:][temp]

        feature_index=0
        new_inqury_output={}
        for feature in targets_list:
            local_max_value=large_error_output[feature_index]*1.1
            local_min_value=large_error_output[feature_index]*0.9

            new_inqury_output[feature]=[local_min_value,local_max_value]
            feature_index=feature_index+1

        suggest_output_bound.append(new_inqury_output)

        feature_index = 0
        new_inqury_point = {}
        for feature in features_list:
            max_value = feature_and_range_dict[feature][1]
            min_value = feature_and_range_dict[feature][0]

            interval = (max_value - min_value) / 100

            local_max_value = large_error_input[feature_index] + 5*interval
            local_min_value = large_error_input[feature_index] - 5*interval

            if local_max_value > max_value:
                local_max_value = max_value
            if local_min_value < min_value:
                local_min_value = min_value

            new_inqury_point[feature]=[local_min_value,local_max_value]
            feature_index += 1

        suggest_input_bound.append(new_inqury_point)



    return [suggest_output_bound, suggest_input_bound]