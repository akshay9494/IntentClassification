from abc import ABC, abstractmethod
from configuration import Configurations

class DataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass

    def prepocess_df(self, df):
        """
        Preprocesses the loaded df to remove duplicates and rows with less than 5 labels
        :param df: loaded dataframe
        :return: dataframe with duplicates and lesser instances removed.
        """
        df.drop_duplicates(inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, how='any', inplace=True)
        label_counts = df['Label'].value_counts().rename('label_counts')
        df = df.merge(label_counts.to_frame(),
                      left_on='Label',
                      right_index=True)
        df = df[df.label_counts >= 5]
        df.drop(columns=['label_counts'], inplace=True)
        return df