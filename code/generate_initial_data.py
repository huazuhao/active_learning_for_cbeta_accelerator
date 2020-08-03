

#updated to remove decision variables with zero range from being a feature
#Generate training data for NN
import paretoutil as PU
#from pisapy import ParetoUtil as PU
import numpy as np
import os
import pickle
import json

#Helper Function
#For making one training point
def MakeIndividual(var_param_dir,FileName,XDic,AddPath,Number=0):
    #The number of individual from Xdic
    BaseFront = PU.GetAFront(var_param_dir, 1, AddPath)
    #BaseFront=PU.GetAFront(1,AddPath='') #Get an example front from output
    for feature in XDic: #replace values
        BaseFront[feature][1]=XDic[feature][Number]
    PU.MakeFrontFileFromFront(FileName,BaseFront,ID_number=0,FileClean=0)


def generate_initial_data(Nsamples):


    #here, I enter the location of where the dcgun file is located
    accelerator_file_location ='/nfs/acc/user/zh296/inopt/examples/dcgun/'

    decdic_location = accelerator_file_location+'optconf/docs/decisions.enxy.qb'
    DecDic=PU.MakeDecDic(decdic_location)
    objdic_location = accelerator_file_location+'optconf/docs/objectives.enxy.qb'
    ObjDic=PU.MakeObjDic(objdic_location)
    condic_location = accelerator_file_location+'optconf/docs/constraints'
    ConDic=PU.MakeConDic(condic_location)

    #Features=['total_charge','sigma_xy','t_ellips','t_slope'] #these decisions will be used to make random inputs for training data
    Features=[feature for feature in DecDic]  		# OR Use this line to make all decisions a feature
    XDic={} #features go here
    RemoveZeroRangeFeatures=1  #don't-remove=0, anyother value will result in removing
    #Make a dictionary with feature values ranging from min/max as given in decisions file
    for feature in Features:
        f_min=DecDic[feature][0]
        f_max=DecDic[feature][1]
        if f_min!=f_max or RemoveZeroRangeFeatures==0:
            XDic[feature]=np.random.uniform(f_min,f_max,Nsamples)

    pickle_out = open("feature_and_range.p","wb")
    pickle.dump(DecDic, pickle_out)
    pickle_out.close()

    #print('xdic is')
    #print(XDic)


    #Now run GPT to get out objectives
    var_param_dir = accelerator_file_location +'optconf/var_param'
    AddPath = accelerator_file_location
    func_eval_location = accelerator_file_location+'zuhao_func_eval.py'
    json_file_location = accelerator_file_location+'TmpInd.json'

    YDic={}#targets go here
    for obj in ObjDic:
        YDic[obj]=[]
    for j in range(Nsamples):
        print('drawing sample j',j)
        FileName=accelerator_file_location+'tmp_ind.txt'
        MakeIndividual(var_param_dir,FileName,XDic,AddPath,j)
        os.system('python '+func_eval_location+' -f '+ FileName+' -s '+json_file_location+' -p '+accelerator_file_location)
        with open(json_file_location,'r') as f:
            TmpDic=json.load(f)
        f.close()
        for obj in ObjDic:
            YDic[obj].append(TmpDic['docs']['objectives'][obj])

    #Save data with pickle

    initial_x_data_name = "X_trainingAll.p"
    initial_y_data_name = "Y_trainingAll.p"
    pickle.dump(XDic,open(initial_x_data_name,"wb"))
    pickle.dump(YDic,open(initial_y_data_name,"wb"))

    return [initial_x_data_name,initial_y_data_name]









