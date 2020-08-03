import numpy as np


class probability_for_a_feature():

    def __init__(self,feature_min,feature_max,spread_parameter=1000):
        self.reject_points_list=[]
        self.feature_max=feature_max
        self.feature_min=feature_min
        self.spread=(self.feature_max-self.feature_min)/spread_parameter
        self.cdf_x_axis=None
        self.cdf_y_axis=None
        self.increment=(self.feature_max-self.feature_min)/10000

    def add_reject_points(self,reject_points):
        if reject_points!=None:
            self.reject_points_list.append(reject_points)
            self.update_probability()
        else:
            self.update_probability()

    def update_probability(self):
        # first, I am going to subtrace the holes from my distribution
        x_axis = []
        y_axis = []
        index = self.feature_min
        while index <= self.feature_max:
            x_axis.append(index)
            temp = self.generate_custom_distribution(index, 0, 1)
            # temp here is the pdf value at this index
            y_axis.append(temp)
            index = index + self.increment

        # second, I need to find the lowest negative points
        # so that I can shift the entire pdf to be above zero
        f_min = min(y_axis)
        if f_min > 0:
            f_min = 0

        # third, I shift the pdf to be above zero everywhere and normalize the distribution
        x_axis = []
        y_axis = []
        index = self.feature_min
        while index <= self.feature_max:
            x_axis.append(index)
            temp = self.generate_custom_distribution(index, f_min, 1)
            # since I have not yet normalized anything, the last argument is 1
            # temp here is the pdf value at this index
            y_axis.append(temp)
            index = index + self.increment
        normalization_factor = sum(y_axis)

        # fourth, I compute the cdf
        # I want to compute the cdf because I want to draw sample by inverting the cdf
        cdf_x_axis = []
        cdf_y_axis = []
        index = self.feature_min
        while index <= self.feature_max:
            cdf_x_axis.append(index)
            if len(cdf_y_axis) == 0:
                cdf_y_axis.append(self.generate_custom_distribution(index, f_min, normalization_factor))
            else:
                cdf_y_axis.append(
                    cdf_y_axis[-1] + self.generate_custom_distribution(index, f_min, normalization_factor))
            index = index + self.increment

        self.cdf_x_axis = cdf_x_axis
        self.cdf_y_axis = cdf_y_axis

    def draw_one_sample(self):
        #the last step is that I draw a sample by
        u = np.random.uniform(0, 1, 1)
        u = u[0]
        # now, find which y_axis this u corresponds
        cdf_y_axis = np.asarray(self.cdf_y_axis) #this y axis runs from 0 to 1
        distance = cdf_y_axis - u
        distance = np.abs(distance)
        find_index = distance == min(distance)
        find_index = find_index * 1
        find_index = np.asarray(find_index)
        find_index = find_index.reshape(-1, )
        np_r = np.nonzero(find_index)
        r = self.cdf_x_axis[np_r[0][0]]
        #the reason I can use this find index method is because I know len(cdf_x_axis)==len(cdf_y_axis)
        return r


    def my_gaussian(self,mean, std, x):
        return np.exp(-(x - mean) ** 2 / (2. * std ** 2)) / np.sqrt(2.0 * np.pi * std ** 2)


    def generate_custom_distribution(self,x, min_value, normalization_factor):
        custom_distribution = 1
        for index in range(0, len(self.reject_points_list)):
            custom_distribution = custom_distribution - self.my_gaussian(self.reject_points_list[index], 2*self.spread, x)
        custom_distribution = custom_distribution - min_value
        custom_distribution = custom_distribution / normalization_factor
        return custom_distribution
