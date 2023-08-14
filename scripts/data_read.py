import pandas as pd
from IPython.core.display import display
import pprint
from sklearn.datasets import load_iris
import seaborn as sns



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
        print('df_iris shape: (%i,%i)' %x.join(y).shape)
        print('-----------------------------------------')
        display(x.join(y).head(5))
        return x, y
    
    def BitcoinPrice():
        train_data = pd.read_csv("data/BitcoinPricePrediction/bitcoin_price_Training - Training.csv")
        test_data = pd.read_csv("data/BitcoinPricePrediction/bitcoin_price_1week_Test - Test.csv")
        print('--------------------------------------------------')
        print('train_data_shape: (%i,%i)' % train_data.shape)
        print('--------------------------------------------------')
        pprint.pprint(list(train_data.columns))
        return train_data, test_data
    
    def flights_seaborn():
        df = sns.load_dataset('flights')
        print('df_raw_shape: (%i,%i)' % df.shape)
        pprint.pprint(list(df.columns))
        return df







