# Date: 2022.07.14, Name: Jongwook Lee, Constitute: Dankook University, E-mail: 72220137@dankook.ac.kr
import pandas as pd

class Reader:

    def __init__(self):
        """This class will be read data with csv file format"""
        self.data = None
        self.col = None
        self.row = None
        self.classes = None
        self.class_names = None

    def read(self, input_path):

        self.data = pd.read_csv(input_path)
        self.col = len(self.data.columns[:-1])
        self.row = len(self.data)
        self.class_names = self.data.iloc[:,-1].unique()
        self.classes = len(self.data.iloc[:,-1].unique())
        return self.data

    def print_data(self):

        print('-'*50)
        print('Raw data')
        print(self.data)
        print('-'*50)

        print('-'*50)
        print('Num. of col: ', self.col)
        print('Num. of row: ', self.row)
        print('Name of classes: ', self.class_names)
        print('Num. of classes: ', self.classes)
        print('-'*50)
