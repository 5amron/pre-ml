# portie.py for reading files to import datasets and also checking the structure of them.


import numpy as np





class CheckDataset(object):

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

        self.sample_num = 0
        self.msg = ""
        self.state = True

        if(self.check_x()):
            if(self.check_y()):
                self.state = True
            else:
                self.state = False
        else:
            self.state = False





    def check_x(self):
        x = self.x_data
        try:
            if(type(x) == np.ndarray): # np array

                if(len(x.shape) != 2):
                    self.msg = "your x data should have 2 dimentions!"
                    return False
                elif(x.shape[0]<5 or x.shape[1]<2):
                    self.msg = "your x data should have more data!!"
                    return False

                self.sample_num = x.shape[0]

                if(np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer)): # it's numeric
                    # now checking size of each row
                    size = len(x[0])
                    for row in x:
                        if(len(row) != size):
                            self.msg = "rows should have same size in your x data!"
                            return False

                    return True
                else: # it's sth else other than : float64, 32, int64 , ...
                    self.msg = "your x data is numpy array but it doesn't consist numeric values like (float64, int64, ...)!"
                    return False



            elif(type(x) == list): # python list

                if(type(x[0]) != list):
                    self.msg = "your x data should have 2 dimentions!"
                    return False

                if(len(x)<5 or len(x[0])<2):
                    self.msg = "your x data should have more data!! and should have 2 dimentions!"
                    return False

                self.sample_num = len(x)

                size = len(x[0])
                for row in x:
                    if(len(row) != size):
                        self.msg = "rows should have same size in your x data!"
                        return False
                    for item in row:
                        if(type(item) == list):
                            self.msg = "your x data should have 2 dimentions!"
                            return False
                        if(not isinstance(item, (int, float))):
                            self.msg = "your x data is python list which consists non-numeric values other than float or int!"
                            return False

                return True


            else:
                self.msg = "your x data should be 2d python list or numpy array!"
                return False

        except Exception as e:
            raise e
            self.msg = "something went wrong with your x data, please check out docs agian!"
            return False





    def check_y(self):
        y = self.y_data

        if(type(y) == np.ndarray): # np array

            if(len(y.shape) != 1):
                self.msg = "your y data should have 1 dimention!"
                return False

            if(len(y)<5):
                self.msg = "your y data should have more data!!"
                return False

            if(len(y) != self.sample_num):
                self.msg = "x and y data should have same number of rows!"
                return False

            if(np.issubdtype(y.dtype, np.floating) or np.issubdtype(y.dtype, np.integer)): # it's numeric
                return True
            else: # it's sth else other than : float64, 32, int64 , ...
                self.msg = "your y data is numpy array but it doesn't consist numeric values like (float64, int64, ...)!"
                return False



        elif(type(y) == list): # python list

            if(len(y)<5):
                self.msg = "your y data should have more data!!"
                return False


            if(len(y) != self.sample_num):
                self.msg = "x and y data should have same number of rows!"
                return False

            for each in y:
                if(type(each) == list):
                    self.msg = "your y data should have 1 dimention!"
                    return False
                elif(not isinstance(each, (int, float))):
                    self.msg = "your x data is python list which consists non-numeric values other than float or int!"
                    return False
                else:
                    return True



        else:
            self.msg = "your x data should be 2d python list or numpy array!"
            return False



    def get_state(self):
        return (self.state, self.msg)




#
