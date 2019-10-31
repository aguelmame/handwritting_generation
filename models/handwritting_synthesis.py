import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dropout,
    Dense, Concatenate,
)

from models.generic_model import HandWriting


class HandwritingSynthesis(HandWriting):

    def __init__(self, n_jobs=1):
        super(HandwritingSynthesis, self).__init__(n_jobs=n_jobs)

    def attention(self, sentence, h1):
        alpha = tf.math.exp(h1[:, :, 0:10]),
        beta = tf.math.exp(h1[:, :, 10, 20]),
        kappa = tf.math.exp(h1[:, :, 20, 30])
        kappa += tf.concat(tf.zeros(1), kappa[:-1])
        w = []
        for u in range(h1.shape[1]):
            w.append(
                sentence
                * tf.reduce_sum(
                    alpha
                    * tf.math.exp(- beta * (kappa - u)**2),
                    axis=2,)
            )

        return tf.reduce_sum(w)

    def _make_model(self):

        strokes = Input((None, 3))
        sentence = Input(57)

        lstm1, previous_state = LSTM(
            400,
            return_sequences=True,
            return_state=True,
            name='h1',
            )(strokes)

        _window_coef = Dense(30)(lstm1)
        window = self.compute_window(sentence, _window_coef)

        _input2 = Concatenate(name='Skip1')([strokes, window, lstm1])
        lstm2 = LSTM(
            400,
            return_sequences=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h2',
        )(_input2)

        _input3 = Concatenate(name='Skip2')([strokes, window, lstm2])
        _input3 = lstm2
        lstm = LSTM(
            400,
            return_sequences=True,
            kernel_regularizer=self.regularizer,
            recurrent_regularizer=self.regularizer,
            name='h3',
            )(_input3)

        lstm = Concatenate(name='Skip3')([lstm1, lstm2, lstm])

        y_hat = Dense(121, name='MixtureCoef')(lstm)
        mixture_coefs = self._mixture_coefs(y_hat)

        model = Model(inputs=strokes, outputs=mixture_coefs)

        if self.compile:
            optimizer = tf.keras.optimizers.RMSprop(
                    lr=self.lr,
                    rho=self.rho,
                    momentum=self.momentum,
                    epsilon=self.epsilon,
                    centered=self.centered,
                    clipnorm=10,
                    clipvalue=5,
                )

            model.compile(
                optimizer,
                loss=self.loss_function,
            )

        return model
