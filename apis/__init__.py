from flask_restplus import Api
from .train_intentions import api as ns1
from .classify_intentions import api as ns2
from .training_status import api as ns3

api = Api(
    title='Intent Classification',
    version='1.0'
    # All API metadatas
)

api.add_namespace(ns1)
api.add_namespace(ns2)
api.add_namespace(ns3)