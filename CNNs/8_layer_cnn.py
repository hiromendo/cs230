from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnnModel(features, labels, mode):

  #based on AlexNet

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 300, 300, 1])

  # Convolutional Layer #1
  # Computes 64 features using a 3x3 filter with ReLU activation.
  # Input Tensor Shape: [m, 300, 300, 1]
  # Output Tensor Shape: [m, 150, 150, 96]
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=96,kernel_size=[2, 2],strides=(2,2),padding="valid",activation=tf.nn.relu)

   # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [m, 300, 300, 96]
  # Output Tensor Shape: [m, 74, 74, 96]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, padding='valid')

  # Convolutional Layer #2
  # Input Tensor Shape: [m, 74, 74, 96]
  # Output Tensor Shape: [m, 70, 70, 256]
  conv2 = tf.layers.conv2d(inputs=pool1,filters=256,kernel_size=[5, 5],strides=(1,1),padding='valid',activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [m, 70, 70, 256]
  # Output Tensor Shape: [m, 34, 34, 256]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

  # Convolutional Layer #3
  # Input Tensor Shape: [m, 34, 34, 256]
  # Output Tensor Shape: [m, 31, 31, 384]
  conv3 = tf.layers.conv2d(inputs=pool2,filters=384,kernel_size=[3, 3],strides=(1,1),padding="valid",activation=tf.nn.relu)

  # Convolutional Layer #4
  # Input Tensor Shape: [m, 31, 31, 384]
  # Output Tensor Shape: [m, 28, 28, 384]
  conv4 = tf.layers.conv2d(inputs=conv3,filters=384,kernel_size=[3, 3],strides=(1,1),padding="valid",activation=tf.nn.relu)

  # Convolutional Layer #6
  # Input Tensor Shape: [m, 28, 28, 384]
  # Output Tensor Shape: [m, 25, 25, 256]
  conv5 = tf.layers.conv2d(inputs=conv4,filters=256,kernel_size=[3, 3],strides=(1,1),padding="valid",activation=tf.nn.relu)

  # Pooling Layer #5
  # Input Tensor Shape: [m, 25, 25, 256]
  # Output Tensor Shape: [m, 12, 12, 256]
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [m, 12, 12, 256]
  # Output Tensor Shape: [m, 12 * 12 * 256]
  pool5_flat = tf.reshape(pool5, [-1, 12 * 12 * 256])

  # Dense Layer # 1
  # Densely connected layer with 4096 neurons
  # Input Tensor Shape: [m, 9 * 9 * 512]
  # Output Tensor Shape: [m, 4096]
  dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)

  # Add dropout operation; 0.8 probability that element will be kept
  dropout1 = tf.layers.dropout(inputs=dense1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer # 1
  # Densely connected layer with 4096 neurons
  # Input Tensor Shape: [m, 4096]
  # Output Tensor Shape: [m, 4096]
  dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu)

  # Add dropout operation; 0.8 probability that element will be kept
  dropout2 = tf.layers.dropout(inputs=dense2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer # 2 (Output)
  # Input Tensor Shape: [m, 4096]
  # Output Tensor Shape: [batch_size, c] #NEEDS TO BE UPDATED WITH APPROPRIATE # OF CLASSES #6
  logits = tf.layers.dense(inputs=dropout2, units=2)

  # Apply Softmax
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load test data
  train_data = np.load("dataset/x_train35K_arabic_german.npy")
  train_labels = np.asarray(np.load("dataset/y_train35K_arabic_german.npy"),dtype=np.int32)
  eval_data = np.load("dataset/x_val7_5K_arabic_german.npy")
  eval_labels = np.asarray(np.load("dataset/y_val7_5K_arabic_german.npy"),dtype=np.int32)
  train_test_data = np.load("dataset/trainx_1K_a_g.npy")
  train_test_labels = np.load("dataset/trainy_1K_a_g.npy")
  # Create the Estimator
  audio_classifier = tf.estimator.Estimator(model_fn=cnnModel, model_dir="checkpoint_path")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #ogging_hook = tf.train.LoggingTensorHook(
  #    tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  audio_classifier.train(
      input_fn=train_input_fn,
      steps=1)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = audio_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()


