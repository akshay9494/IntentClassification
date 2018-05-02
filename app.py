from flask import Flask
from apis import api
import logging
import os

logging.getLogger(__file__)
logdir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
if not os.path.isdir(logdir):
    os.makedirs(logdir)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
logname = 'logs.log'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)#, filename=os.path.join(logdir, logname))

app = Flask(__name__)
api.init_app(app)

app.run(debug=True)