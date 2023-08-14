import pandas as pd
from sklearn.datasets import load_iris



class My_Data_Read:
    def SIG_mpg():
        train_data = pd.read_table("data/SIGNATE_exercises_mpg/train.tsv")
        test_data = pd.read_table("data/SIGNATE_exercises_mpg/test.tsv")
        return train_data, test_data
    
    def RedWineQuality():
        train_data = pd.read_csv('data/RedWineQuality/train.csv')
        test_data = pd.read_csv('data/RedWineQuality/test.csv')
        return train_data, test_data
    
    def Iris():
        dataset = load_iris()
        x = pd.DataFrame(dataset.data,columns=dataset.feature_names)
        y = pd.Series(dataset.target, name='Variety')
        print('----------------------------------------')
        print('df_iris shape: (%i,%i)' %X.join(y).shape)
        print('-----------------------------------------')
        display(x.join(y).head(5))()
        return x, y











