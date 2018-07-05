from flask_restplus import Namespace, Resource, fields
import logging
from core.IntentClassification.IntentClassifier import IntentClassifier
import tensorflow as tf
from log4mongo.handlers import MongoHandler
from core.IntentClassification.configuration.Configurations import Configurations

config = Configurations()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(MongoHandler(host=config.log_properties.host, collection=config.log_properties.col_name))

graph = None
intent_classification_instance = None

api = Namespace('Classify Intentions', description='Intent Classification APIs.')


classification_payload = api.model('classify', {
    'sentence': fields.String(required=True, description='sentence to get classification for from the model.')
})

classification_response = api.model('Classification Response', {
    'intent': fields.String(description='Intent of the sentence')
})

@api.route('/initialize')
class Initialize(Resource):
    """Initialize model for intent classification"""
    # @api.doc('basic mathematical computations')
    # @api.expect(classification_payload)
    # @api.marshal_with(classification_response, code=200)
    def get(self):
        """Initialize model for intent classification"""
        global intent_classification_instance
        global graph
        logger.info('Received request')
        intent_classification_instance = IntentClassifier()
        graph = tf.get_default_graph()
        return 'Success', 200


@api.route('/isInitialized')
class IsInitialized(Resource):
    """Checks if the model is initialized and ready to classify."""
    # @api.doc('basic mathematical computations')
    # @api.expect(classification_payload)
    # @api.marshal_with(classification_response, code=200)
    def get(self):
        """Checks if the model is initialized and ready to classify."""
        global intent_classification_instance
        if intent_classification_instance is None:
            return 'Not Initialized'
        else:
            return 'Initialized', 200


@api.route('/classify')
class Classify(Resource):
    """Intent Classification of sentences."""
    # @api.doc('basic mathematical computations')
    @api.expect(classification_payload)
    @api.marshal_with(classification_response, code=200)
    def post(self):
        """Intent Classification of sentences."""
        logger.info('Received request')
        sentence = self.api.payload['sentence']
        global intent_classification_instance
        global graph
        if intent_classification_instance is None:
            return 'Model uninitialized. Please initialize.'
        with graph.as_default():
            intent = intent_classification_instance.get_intent(sentence)
        return {'intent': intent}, 200



        