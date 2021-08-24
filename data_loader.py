import os

import pandas as pd

root_dir = os.getcwd()
data_path = os.path.join(root_dir, 'dataset')

class DataLoader():
    
    def __init__(self,file_name):
        
        self.file_name = file_name
        if not os.path.exists(os.path.join(data_path, 'quora_data.csv')):
            raise ValueError("Dataset is not available in the desired location.")

        else:   
            pass

    
    def load_data(self):

        file_path = os.path.join(data_path, self.file_name)
        quora_data = pd.read_csv(file_path)
        return quora_data


obj = DataLoader(file_name = 'quora_data.csv')
print(obj.load_data())