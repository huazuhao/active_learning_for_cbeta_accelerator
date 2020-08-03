import numpy as np

class per_fold_object():
    def __init__(self,training_set,validation_set):
        self.train_index=training_set
        self.val_index=validation_set


def generating_k_fold(k_fold,x_data,y_data):

    total_data_x = x_data

    # what I do now is that I divide the data into k folds of data
    ints = np.random.permutation(np.arange(0, total_data_x.shape[0], 1, dtype=int))

    fold_list = []

    k_fold = k_fold
    percent = 1.0 / k_fold
    for index in range(0, k_fold):
        begin_index = np.int(total_data_x.shape[0] * percent * index)
        finish_index = np.int(total_data_x.shape[0] * percent * (index + 1))

        ints_test = ints[begin_index: finish_index]

        ints_train = []
        for i in range(0, total_data_x.shape[0]):
            if not i in ints_test:
                ints_train.append(i)
        ints_train = np.asarray(ints_train)

        fold_list.append(per_fold_object(ints_train, ints_test))

    return fold_list