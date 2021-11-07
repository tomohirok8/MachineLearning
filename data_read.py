import pandas as pd



class My_Data_Read:
    def SIG_mpg():
        train_data = pd.read_table("data/SIGNATE_exercises_mpg/train.tsv")
        test_data = pd.read_table("data/SIGNATE_exercises_mpg/test.tsv")
        
        return train_data, test_data
    
    def RedWineQuality():
        x_train = pd.read_csv('data/RedWineQuality/train.csv')
        x_test = pd.read_csv('data/RedWineQuality/test.csv')
        
        return x_train, x_test











