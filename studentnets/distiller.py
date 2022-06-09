# Class that distills the knowledge from a given teacher model to a given student model.

import numpy as np

from tensorflow import keras


class Distiller(keras.Model):
    """Train the student using teacher information through knowledge distillation.
    Details of this process are in explained http://arxiv.org/abs/1503.02531.

    Args:
        student: Student network, usually small and basic.
        teacher: Teacher network, usually big.
    """

    def __init__(self, student: keras.Model, teacher: keras.Model):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer: keras.optimizers,
        metrics: keras.metrics,
        student_loss_fn: callable,
        distillation_loss_fn: callable,
        alpha: float = 0.1,
        temperature: int = 3,
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
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data: np.ndarray):
        """Train the student network through one feed forward."""
        x, y = data

        loss = self.__compute_loss(x, y)

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )

        return results

    def __compute_loss(self, x: np.ndarray, y: np.ndarray):

        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

    def test_step(self, data: np.ndarray):
        """Test the student network."""
        x, y = data
        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)

        self.compiled_metrics.update_state(y, y_prediction)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})

        return results
