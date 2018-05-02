from flask_restplus import Api
from .intentions import api as ns1


api = Api(
    title='Intent Classification',
    version='1.0'
    # All API metadatas
)

api.add_namespace(ns1)