from flask import Flask
from apis import api
import logging
import os
from waitress import serve
from log4mongo.handlers import MongoHandler
from core.IntentClassification.configuration.Configurations import Configurations

config = Configurations()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(MongoHandler(host=config.log_properties.host, collection=config.log_properties.col_name))

# logging.getLogger(__file__)
# logdir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
# if not os.path.isdir(logdir):
#     os.makedirs(logdir)
# LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
#               '-35s %(lineno) -5d: %(message)s')
# logname = 'logs.log'
# logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)#, filename=os.path.join(logdir, logname))

logger.info('Beginning')

app = Flask(__name__)
api.init_app(app)

# app.run(debug=False, threaded=True)
serve(app, host='0.0.0.0', port=9494)