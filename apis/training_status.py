from flask_restplus import Namespace, Resource
import logging
from core.IntentClassification.IntentTrainer import train
from log4mongo.handlers import MongoHandler
from core.IntentClassification.configuration.Configurations import Configurations
import pymongo


config = Configurations()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(MongoHandler(host=config.log_properties.host, collection=config.log_properties.col_name))

from pymongo import MongoClient
client = MongoClient(config.log_properties.host, config.log_properties.port)
db = client[config.log_properties.db_name]
col = db[config.log_properties.col_name]

# print(col.find_one())

graph = None
intent_classification_instance = None

api = Namespace('Training Status', description='Training Status APIs.')

@api.route('/training_status')
class TrainingStatus(Resource):
    """Train the model for intent classification."""
    # @api.doc('basic mathematical computations')
    # @api.expect(s2me_payload)
    # @api.marshal_with(s2me_response, code=200)
    def get(self):
        """Training Status for intent classification."""
        list_to_return = []
        for c in col.find({'level': 'INFO'}).sort('timestamp', pymongo.DESCENDING).limit(10):
            list_to_return.append(c['message'])
        return list_to_return, 200


