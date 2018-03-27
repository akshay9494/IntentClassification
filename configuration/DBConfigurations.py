class DBConfigurations:
    def __init__(self, db_config):
        self.db_name = db_config['DBName']
        self.collection_name = db_config['collectionName']
        self.training_status_collection = db_config['trainingStatusCollection']