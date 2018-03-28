from keras.models import Model
from keras import optimizers
from keras.layers import Embedding, Dropout, LSTM, Concatenate, Dense, \
    BatchNormalization, GlobalMaxPooling1D, Input, concatenate


class ModelArchitectures:
    def __init__(self, model_properties):
        self.model_properties = model_properties

    def _model_lstm(self, embedding_matrix, word_index, num_classes):
        embedding_layer = Embedding(len(word_index)+1,
                                    embedding_matrix.shape[1],
                                    weights=[embedding_matrix],
                                    input_length=self.model_properties.max_length,
                                    trainable=False)

        sequence_input = Input(shape=(self.model_properties.max_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Dropout(self.model_properties.dropout)(embedded_sequences)
        x = LSTM(100)(x)
        x = Dropout(self.model_properties.dropout)(x)
        preds = Dense(num_classes, activation='softmax')(x)

        model = Model(sequence_input, preds)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=self.model_properties.learning_rate),
                      metrics=['acc'])
        return model


    def _model_lstm_tfidf(self, embedding_matrix, word_index, input_shape, num_classes):
        embedding_layer = Embedding(len(word_index) + 1,
                                    embedding_matrix.shape[1],
                                    weights=[embedding_matrix],
                                    input_length=self.model_properties.max_length,
                                    trainable=False)

        sequence_input = Input(shape=(self.model_properties.max_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Dropout(self.model_properties.dropout)(embedded_sequences)
        x = BatchNormalization()(x)
        x = LSTM(100)(x)
        x = Dropout(self.model_properties.dropout)(x)

        tfidf_input = Input(shape=(input_shape,), dtype='float')
        y = Dense(embedding_matrix.shape[1], kernel_initializer='glorot_uniform', activation='tanh')(tfidf_input)
        y = Dropout(self.model_properties.dropout)(y)
        y = BatchNormalization()(y)
        y = Dense(embedding_matrix.shape[1], kernel_initializer='glorot_uniform', activation='tanh')(y)
        y = Dropout(self.model_properties.dropout)(y)

        merge_layer = concatenate([x, y])
        merge_layer = BatchNormalization()(merge_layer)
        output_layer = Dense(num_classes, activation='softmax')(merge_layer)

        model = Model([sequence_input, tfidf_input], output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Nadam(lr=self.model_properties.learning_rate),
                      metrics=['acc'])
        return model

    def get_model(self):
        model = {
            'lstm': self._model_lstm,
            'lstm_tfidf': self._model_lstm_tfidf
        }
        return model[self.model_properties.model_identifier]