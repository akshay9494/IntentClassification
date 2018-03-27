from .DataLoader import DataLoader

class DBDataLoader(DataLoader):
    def __init__(self, db_properties):
        self.db_properties = db_properties

    def load_data(self):
        pass
