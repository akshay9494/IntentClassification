from flask_restplus import Namespace, Resource
import logging
from core.IntentClassification.IntentTrainer import train
from log4mongo.handlers import MongoHandler
from core.IntentClassification.configuration.Configurations import Configurations

config = Configurations()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(MongoHandler(host=config.log_properties.host, collection=config.log_properties.col_name))

graph = None
intent_classification_instance = None

api = Namespace('Train Intentions', description='Train Intents APIs.')

@api.route('/train')
class Train(Resource):
    """Train the model for intent classification."""
    # @api.doc('basic mathematical computations')
    # @api.expect(s2me_payload)
    # @api.marshal_with(s2me_response, code=200)
    def get(self):
        """Train the model for intent classification."""
        logger.info('Received request')
        train()
        return 'Training complete.', 200


