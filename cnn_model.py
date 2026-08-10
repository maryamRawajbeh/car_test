# -*- coding: utf-8 -*-
r"""
===================================================================
Shared CNN architecture pieces (custom layer + focal loss + builder)
===================================================================
Used by both train_cnn.py (to build/train the model) and predict.py (which
must import this module -- even if it never calls build_cnn() directly --
so tf.keras.models.load_model() can find these registered custom classes by
name when deserializing a saved cnn_model.keras. Without this shared,
explicitly-registered module, loading a model that used a raw
`layers.Lambda(lambda ...)` (or an unregistered custom loss) would either
fail outright under Keras 3's safe-mode deserialization, or require every
loader to redefine byte-identical classes by hand -- both fragile.
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers, models


@keras.saving.register_keras_serializable(package="car_test", name="FrequencyAveragePooling")
class FrequencyAveragePooling(layers.Layer):
    """Averages out the frequency axis (axis=1) of a (batch, freq, time, channels)
    log-mel feature map, leaving a (batch, time, channels) sequence that a
    recurrent layer can read temporal order from -- lets the model see HOW a
    sound evolves over the clip (e.g. intermittent clicks vs a continuous
    squeal) instead of only a single frequency-and-time-agnostic summary."""

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3])


@keras.saving.register_keras_serializable(package="car_test", name="SparseCategoricalFocalLoss")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Focal loss (Lin et al., 2017) for sparse integer labels:
        loss = -(1 - p_t)^gamma * log(p_t)
    where p_t is the predicted probability of the TRUE class. Down-weights
    samples the model already classifies confidently/correctly and focuses
    training on the hard ones -- complements (does not replace) the
    class_weight="balanced" rebalancing passed separately to model.fit,
    which only corrects for class FREQUENCY, not per-example difficulty.
    Returns one loss value per sample (no reduction inside call()), so
    Keras' own class_weight/sample_weight handling in model.fit still
    applies on top of it exactly like it would for the built-in loss."""

    def __init__(self, gamma=2.0, name="sparse_categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        true_class_probs = tf.gather(y_pred, y_true, batch_dims=1)
        cross_entropy = -tf.math.log(true_class_probs)
        modulating_factor = tf.pow(1.0 - true_class_probs, self.gamma)
        return modulating_factor * cross_entropy

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config


def build_cnn(input_shape, n_classes, use_temporal_head=True, use_focal_loss=True, focal_gamma=2.0,
              learning_rate=1e-3):
    """A small CNN, deliberately simple to keep training time manageable on a
    laptop CPU. `use_temporal_head=True` adds a small BiGRU (via
    FrequencyAveragePooling) after the conv blocks instead of pooling
    straight to GlobalAveragePooling2D."""
    conv_layers = [
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
    ]

    if use_temporal_head:
        pooling_layers = [
            FrequencyAveragePooling(name="freq_pool"),
            layers.Bidirectional(layers.GRU(32)),
        ]
    else:
        pooling_layers = [layers.GlobalAveragePooling2D()]

    head_layers = [
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ]

    model = models.Sequential(conv_layers + pooling_layers + head_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=SparseCategoricalFocalLoss(gamma=focal_gamma) if use_focal_loss else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
