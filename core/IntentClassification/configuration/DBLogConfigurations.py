class DBLogConfigurations:
    def __init__(self, log_config):
        self.host = log_config['host']
        self.port = log_config['port']
        self.db_name = log_config['dbName']
        self.col_name = log_config['colName']