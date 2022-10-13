# Class that distills the knowledge from a given teacher model to a given student model.

import numpy as np
import operator

import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow import keras

from . import util


class Distiller(keras.Model):
    """Train the student using teacher information through knowledge distillation.

    Args:
        student: Student network, usually small and basic.
        teacher: Teacher network, usually big.

    Reference paper: http://arxiv.org/abs/1503.02531.
    """

    def __init__(self, student: keras.Model, teacher: keras.Model):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(
        self,
        optimizer: str,
        student_loss_fn: callable,
        distill_loss_fn: callable,
        alpha: float = 0.1,
        temperature: int = 10,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights.
            metrics: Keras metrics for evaluation.
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth.
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions.
            alpha: Weight to student_loss_fn and 1-alpha to distillation_loss_fn.
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, run_eagerly=True)
        self.student_loss_fn = util.choose_loss(student_loss_fn)
        self.distillation_loss_fn = util.choose_loss(distill_loss_fn)
        self.alpha = alpha
        self.temperature = temperature

        self.student_loss_track = tf.keras.metrics.Mean("student_loss")
        self.distill_loss_track = tf.keras.metrics.Mean("distill_loss")
        self.loss_track = tf.keras.metrics.Mean("loss")
        self.categorical_accuracy = tf.keras.metrics.CategoricalAccuracy("acc")

    def train_step(self, data):
        """Train the student network through one feed forward."""
        x, y = data
        x_teacher = self.tf_shuffle_axis(x, 1)
        teacher_predictions = self.teacher(x_teacher, training=False)

        # Select only the first 16 features and (p_T, eta, phi) for the student.
        x_student = tf.stack([x[:, :16, 5], x[:, :16, 8], x[:, :16, 11]], 2)
        x_student = self.tf_shuffle_axis(x_student, 1)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x_student, training=True)

            student_loss = tf.reduce_mean(self.student_loss_fn(y, student_predictions))
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.categorical_accuracy.update_state(y, student_predictions)
        self.student_loss_track.update_state(student_loss)
        self.distill_loss_track.update_state(distillation_loss)
        self.loss_track.update_state(loss)
        results = {m.name: m.result() for m in self.metrics}

        return results

    @property
    def metrics(self):
        """Metrics to be reseted at each epoch."""
        return [
            self.student_loss_track,
            self.distill_loss_track,
            self.loss_track,
            self.categorical_accuracy,
        ]

    def test_step(self, data: np.ndarray):
        """Test the student network."""
        x, y = data

        # Select only the first 16 features and (p_T, eta, phi) for the student.
        x = tf.stack([x[:, :16, 5], x[:, :16, 8], x[:, :16, 11]], 2)
        x = self.tf_shuffle_axis(x, 1)

        y_prediction = self.student(x, training=False)
        student_loss = tf.reduce_mean(self.student_loss_fn(y, y_prediction))

        self.student_loss_track.update_state(student_loss)
        self.categorical_accuracy.update_state(y, y_prediction)

        results = {
            "student_loss": self.student_loss_track.result(),
            "acc": self.categorical_accuracy.result(),
        }

        return results

    def tf_shuffle_axis(self, value, axis=0, seed=None, name=None):
        """Same as tf.shuffle but takes an axis."""
        perm = list(range(tf.rank(value)))
        perm[axis], perm[0] = perm[0], perm[axis]
        value = tf.random.shuffle(tf.transpose(value, perm=perm))
        value = tf.transpose(value, perm=perm)
        return value
