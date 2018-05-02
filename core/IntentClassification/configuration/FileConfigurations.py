class FileConfigurations:
    def __init__(self, file_config):
        self.path = file_config['path']
        self.data_column = file_config['dataColumn']
        self.label_column = file_config['labelColumn']
        self.file_type = file_config['fileType']
