from keras.callbacks import Callback
import logging
from log4mongo.handlers import MongoHandler
from core.IntentClassification.configuration.Configurations import Configurations

config = Configurations()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(MongoHandler(host=config.log_properties.host, collection=config.log_properties.col_name))


class LogEpochStats(Callback):
    def __init__(self):
        super(LogEpochStats, self).__init__()
        # self.steps_per_epoch = steps_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        logger.info('Epoch {} started.'.format(epoch + 1))

    # def on_batch_end(self, batch, logs=None):
    #     logs = logs or {}
    #     logger.info('Batch {}/{}, \tAccuracy -> {}, \t'
    #                 'Loss -> {}'
    #                 .format(batch, self.steps_per_epoch, logs.get('acc'), logs.get('loss')))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info('Epoch {} ended.'.format(epoch + 1))
        logger.info('Train accuracy -> {}, Train Loss -> {}, Validation Accuracy -> {}, Validation loss -> {}'
                    .format(logs.get('acc'), logs.get('loss'), logs.get('val_acc'), logs.get('val_loss')))
