from .DataLoader import DataLoader
import pandas as pd
import logging

class FileDataLoader(DataLoader):
    def __init__(self, file_properties):
        self.file_properties = file_properties


    def load_data(self):
        """
        loads the data from the csv path supplied
        :return: dataframe consisting of two columns named 'Text' and 'Data'
        """
        logging.info('Loading Data from CSV.')
        df = pd.read_csv(self.file_properties.path, encoding='iso-8859-1')
        df = df[[self.file_properties.data_column, self.file_properties.label_column]]
        df.rename(columns={self.file_properties.data_column: 'Text', self.file_properties.label_column: 'Label'},
                  inplace=True)
        df = self.prepocess_df(df)
        return df
