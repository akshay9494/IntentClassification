from .FileConfigurations import FileConfigurations
from .DBConfigurations import DBConfigurations

class DataSourceConfigurations:
    def __init__(self, data_source_properties):
        # self.data_source_properties = data_source_properties
        if 'file' in data_source_properties.keys():
            self.file_properties = FileConfigurations(data_source_properties['file'])
        else:
            self.db_properties = DBConfigurations(data_source_properties['DB'])