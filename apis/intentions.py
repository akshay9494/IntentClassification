from flask_restplus import Namespace, Resource, fields
import os
import logging
from core.IntentClassification.IntentClassifier import IntentClassifier
from core.IntentClassification.IntentTrainer import train
from flask import abort

intent_classification_instance = None

api = Namespace('Intentions', description='Intent Classification APIs.')

classification_payload = api.model('classify', {
    'sentence': fields.String(required=True, description='sentence to get classification for from the model.')
})

classification_response = api.model('Classification Response', {
    'intent': fields.String(description='Intent of the sentence')
})

@api.route('/train')
class Train(Resource):
    """Train the model for intent classification."""
    # @api.doc('basic mathematical computations')
    # @api.expect(s2me_payload)
    # @api.marshal_with(s2me_response, code=200)
    def get(self):
        """Train the model for intent classification."""
        logging.info('Received request')
        train()
        return 'Training complete.', 200


@api.route('/initialize')
class Initialize(Resource):
    """Initialize model for intent classification"""
    # @api.doc('basic mathematical computations')
    # @api.expect(classification_payload)
    # @api.marshal_with(classification_response, code=200)
    def get(self):
        """Initialize model for intent classification"""
        global intent_classification_instance
        logging.info('Received request')
        intent_classification_instance = IntentClassifier()
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
        logging.info('Received request')
        sentence = self.api.payload['sentence']
        global intent_classification_instance
        if intent_classification_instance is None:
            return 'Model uninitialized. Please initialize.'
        intent = intent_classification_instance.get_intent(sentence)
        return {'intent': intent}, 200