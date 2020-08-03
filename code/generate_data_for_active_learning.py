

import paretoutil as PU
import numpy as np
import os
import pickle
import json
from probability import probability_for_a_feature
import time

######Helper Function###########
def MakeIndividual(var_param_dir,FileName,XDic,AddPath,Number=0):
    BaseFront = PU.GetAFront(var_param_dir, 1, AddPath)
    for feature in XDic: #replace values
        BaseFront[feature][1]=XDic[feature][Number]
    PU.MakeFrontFileFromFront(FileName,BaseFront,ID_number=0,FileClean=0)


def active_learning_query_point(suggest_output_bound, suggest_input_bound, threshold_cut, num_sample_for_each_bad_point, num_of_correction_points, iteration_counter):
    print('begin to generate new sample based on the idea of active learning')
    #the argument new_input_dict should be the end result of the function
    #finding_worst_prediction_and_return_suggested_inqury_points

    XDic = {}
    YDic = {}

    accelerator_file_location = '/nfs/acc/user/zh296/inopt/examples/dcgun/'

    decdic_location = accelerator_file_location + 'optconf/docs/decisions.enxy.qb'
    DecDic = PU.MakeDecDic(decdic_location)
    objdic_location = accelerator_file_location + 'optconf/docs/objectives.enxy.qb'
    ObjDic = PU.MakeObjDic(objdic_location)
    condic_location = accelerator_file_location + 'optconf/docs/constraints'
    ConDic = PU.MakeObjDic(condic_location)

    ######arguments for GPT######
    var_param_dir = accelerator_file_location +'optconf/var_param'
    AddPath = accelerator_file_location
    func_eval_location = accelerator_file_location+'zuhao_func_eval.py'
    json_file_location = accelerator_file_location+'TmpInd.json'


    for inquiry_index in range(0,len(suggest_input_bound)):
        input_bound_dict = suggest_input_bound[inquiry_index]
        output_bound_dict = suggest_output_bound[inquiry_index]

        #here, I need to initialize the list of probabilities
        probability_list=[]
        for feature in input_bound_dict.keys():
            input_min = input_bound_dict[feature][0]
            input_max = input_bound_dict[feature][1]
            temp = probability_for_a_feature(input_min,input_max)
            temp.add_reject_points(None)
            probability_list.append(temp)

        success_sample = 0
        timer = int(time.time())
        while success_sample < num_sample_for_each_bad_point:

            current_time = int(time.time())
            if current_time-timer>3600*2:
                break
                #if after two hours of trying, I still cannot
                #gnerated desired number of fixing point for this particular bad points, I am just going to be happy with
                #however many fixing points I have

            try_dic={}
            temp_index=0
            for feature in input_bound_dict.keys():
                temp = probability_list[temp_index]
                temp = temp.draw_one_sample()
                temp = np.asarray([temp])
                try_dic[feature]=temp
                temp_index=temp_index+1

            #now, after generating the input file, I will actually generate the output file based on GPT
            try_output_dict={}
            for feature in output_bound_dict.keys():
                try_output_dict[feature]=[]
            FileName = accelerator_file_location + 'tmp_ind.txt'
            MakeIndividual(var_param_dir, FileName, try_dic, AddPath)
            os.system(
                'python ' + func_eval_location + ' -f ' + FileName + ' -s ' + json_file_location + ' -p ' + accelerator_file_location)
            with open(json_file_location, 'r') as f:
                TmpDic = json.load(f)
            f.close()
            for obj in ObjDic:
                try_output_dict[obj].append(TmpDic['docs']['objectives'][obj])

            #now, I am going to see whether the generated datapoint from GPT fits the output bound
            keep_this_sample=True
            for feature in output_bound_dict.keys():
                output_min = output_bound_dict[feature][0]
                output_max = output_bound_dict[feature][1]
                threshold_min = threshold_cut[feature][0]
                threshold_max = threshold_cut[feature][1]

                actual_output = try_output_dict[feature][0]
                print('the selection criteria is min',max(output_min,threshold_min),'max',min(output_max,threshold_max))
                print('the actual result is',actual_output)
                if actual_output>=max(output_min,threshold_min) and actual_output<=min(output_max,threshold_max):
                    pass
                else:
                    keep_this_sample = False
                    #not only do I not keep this sample
                    #I also need to update the feature probability distribution
                    temp_index=0
                    for temp_feature in try_dic.keys():
                        reject_point = try_dic[temp_feature][0]
                        temp = probability_list[temp_index]
                        temp.add_reject_points(reject_point)
                        temp_index = temp_index+1
                if keep_this_sample==False:
                    break

            if keep_this_sample == True:
                #keep the x point
                for feature in input_bound_dict.keys():
                    new_input = try_dic[feature]
                    try:
                        new_inquiry_point_feature_entry = XDic[feature]
                        temp = np.concatenate((new_inquiry_point_feature_entry, new_input))
                        XDic[feature] = temp
                    except:
                        XDic[feature]=new_input

                #keep the y point
                for feature in output_bound_dict.keys():
                    new_output = try_output_dict[feature]
                    try:
                        new_inquiry_point_output = YDic[feature]
                        temp = np.concatenate((new_inquiry_point_output,new_output))
                        YDic[feature]=temp
                    except:
                        YDic[feature]=new_output
                success_sample = success_sample+1
                print('successfully generated an active learn point and we have finished ',success_sample/num_sample_for_each_bad_point)
                print('and is working on point',inquiry_index)


    #after doing the targeted learning, I will generate random points

    for key in XDic:
        temp = XDic[key]
        Nsamples = num_sample_for_each_bad_point*num_of_correction_points*2-len(temp)
        break

    # Features=['total_charge','sigma_xy','t_ellips','t_slope'] #these decisions will be used to make random inputs for training data
    Features = [feature for feature in DecDic]  # OR Use this line to make all decisions a feature
    randomXDic = {}  # features go here
    RemoveZeroRangeFeatures = 1  # don't-remove=0, anyother value will result in removing
    ###Make a dictionary with feature values ranging from min/max as given in decisions file
    for feature in Features:
        f_min = DecDic[feature][0]
        f_max = DecDic[feature][1]
        if f_min != f_max or RemoveZeroRangeFeatures == 0:
            randomXDic[feature] = np.random.uniform(f_min, f_max, Nsamples)

    randomYDic = {}  # targets go here
    for obj in ObjDic:
        randomYDic[obj] = []
    for j in range(Nsamples):
        print('drawing sample j', j)
        FileName =accelerator_file_location+'tmp_ind.txt'
        MakeIndividual(var_param_dir, FileName, randomXDic, AddPath,j)
        os.system(
            'python ' + func_eval_location + ' -f ' + FileName + ' -s ' + json_file_location + ' -p ' + accelerator_file_location)
        with open(json_file_location, 'r') as f:
            TmpDic = json.load(f)
        f.close()
        for obj in ObjDic:
            randomYDic[obj].append(TmpDic['docs']['objectives'][obj])


    #now, I put the targeted sample points and random sample together
    for feature in XDic.keys():
        xdic_input = XDic[feature]
        randomxdic_input = randomXDic[feature]
        temp = np.concatenate((xdic_input, randomxdic_input))
        XDic[feature] = temp
    for feature in YDic.keys():
        ydic_input = YDic[feature]
        randomydic_input = randomYDic[feature]
        temp = np.concatenate((ydic_input,randomydic_input))
        YDic[feature] = temp

    ###Save data with pickle
    new_x_data_name = "X_active_learning_"+str(iteration_counter)+".p"
    new_y_data_name = "Y_active_learning_"+str(iteration_counter)+".p"
    pickle.dump(XDic, open(new_x_data_name, "wb"))
    pickle.dump(YDic, open(new_y_data_name, "wb"))

    return [new_x_data_name,new_y_data_name]