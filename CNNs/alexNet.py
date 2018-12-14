from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def alexNet(features, labels, mode):

  #based on AlexNet

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 300, 300, 1])

  # Convolutional Layer #1
  # Computes 64 features using a 3x3 filter with ReLU activation.
  # Input Tensor Shape: [m, 300, 300, 1]
  # Output Tensor Shape: [m, 300, 300, 64]
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Convolutional Layer #2
  # Input Tensor Shape: [m, 300, 300, 64]
  # Output Tensor Shape: [m, 300, 300, 64]
  conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)
  
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [m, 300, 300, 64]
  # Output Tensor Shape: [m, 150, 150, 64]
  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='valid')

  # Convolutional Layer #3
  # Input Tensor Shape: [m, 150, 150, 64]
  # Output Tensor Shape: [m, 150, 150, 128]
  conv3 = tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Convolutional Layer #4
  # Input Tensor Shape: [m, 150, 150, 128]
  # Output Tensor Shape: [m, 150, 150, 128]
  conv4 = tf.layers.conv2d(inputs=conv3,filters=128,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [m, 150, 150, 128]
  # Output Tensor Shape: [m, 75, 75, 128]
  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  # Convolutional Layer #5
  # Input Tensor Shape: [m, 75, 75, 128]
  # Output Tensor Shape: [m, 75, 75, 256]
  conv5 = tf.layers.conv2d(inputs=pool2,filters=256,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Convolutional Layer #6
  # Input Tensor Shape: [m, 75, 75, 256]
  # Output Tensor Shape: [[m, 75, 75, 256]
  conv6 = tf.layers.conv2d(inputs=conv5,filters=256,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Convolutional Layer #7
  # Input Tensor Shape: [m, 75, 75, 256]
  # Output Tensor Shape: [m, 75, 75, 256]
  conv7 = tf.layers.conv2d(inputs=conv6,filters=256,kernel_size=[5, 5],strides=(2,2),padding="same",activation=tf.nn.relu)

  # Pooling Layer #3
  # Input Tensor Shape: [m, 75, 75, 256]
  # Output Tensor Shape: [m, 37, 37, 256]
  pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

  # Convolutional Layer #8
  # Input Tensor Shape: [m, 37, 37, 256]
  # Output Tensor Shape: [m, 37, 37, 512]
  conv8 = tf.layers.conv2d(inputs=pool3,filters=512,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Convolutional Layer #9
  # Input Tensor Shape: [m, 37, 37, 512]
  # Output Tensor Shape: [m, 37, 37, 512]
  conv9 = tf.layers.conv2d(inputs=conv8,filters=512,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Convolutional Layer #10
  # Input Tensor Shape: [m, 37, 37, 512]
  # Output Tensor Shape: [m, 37, 37, 512]
  conv10 = tf.layers.conv2d(inputs=conv9,filters=512,kernel_size=[5, 5],strides=(2,2),padding="same",activation=tf.nn.relu)

  # Pooling Layer #4
  # Input Tensor Shape: [m, 37, 37, 512]
  # Output Tensor Shape: [m, 18, 18, 512]
  pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

  # Convolutional Layer #11
  # Input Tensor Shape: [m, 18, 18, 512]
  # Output Tensor Shape: [m, 18, 18, 512]
  conv11 = tf.layers.conv2d(inputs=pool4,filters=512,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Convolutional Layer #12
  # Input Tensor Shape: [m, 18, 18, 512]
  # Output Tensor Shape: [m, 18, 18, 512]
  conv12 = tf.layers.conv2d(inputs=conv11,filters=512,kernel_size=[3, 3],strides=(1,1),padding="same",activation=tf.nn.relu)

  # Pooling Layer #5
  # Input Tensor Shape: [m, 18, 18, 512]
  # Output Tensor Shape: [m, 9, 9, 512]
  pool5 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [m, 9, 9, 512]
  # Output Tensor Shape: [m, 9 * 9 * 512]
 # pool5_flat = tf.reshape(pool5, [-1, 9 * 9 * 512])
  pool5_flat = tf.reshape(pool5, [-1, 2 * 2 * 512])
  # Dense Layer # 1
  # Densely connected layer with 4096 neurons
  # Input Tensor Shape: [m, 8 * 8 * 512]
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
  logits = tf.layers.dense(inputs=dropout2, units=6)

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  #print('confusion matrix:', confusion_matrix(labels, predictions["classes"]), sep='\n')
 # with tf.Session() as session:
     # p = session.run(output, feed_dict={x: eval_data})
     # p_labels = (p > 0.5).astype(int)
     # print('confusion matrix:', confusion_matrix(eval_labels, p_labels), sep='\n')
  confusion_matrix = tf.confusion_matrix(labels, predictions["classes"])
  print(confusion_matrix)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  train_data = np.load("dataset/x_train105K_6lang.npy")
  train_labels = np.asarray(np.load("dataset/y_train105K_6lang.npy"),dtype=np.int32)
  eval_data = np.load("dataset/x_val22_5K_6lang.npy")
  eval_labels = np.asarray(np.load("dataset/y_val22_5K_6lang.npy"),dtype=np.int32)

  test_data = np.load("dataset/x_test22_5K_6lang.npy")
  test_labels = np.asarray(np.load("dataset/y_test22_5K_6lang.npy"),dtype=np.int32)
  # Create the Estimator
  audio_classifier = tf.estimator.Estimator(model_fn=alexNet, model_dir="alexNet")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=50,
      num_epochs=None,
      shuffle=True)
  audio_classifier.train(
      input_fn=train_input_fn,
      steps=1)
     # hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=50,
      num_epochs=1,
      shuffle=False)
  eval_results = audio_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

 # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
 #     x={"x": eval_data},
 #     y=eval_labels,
 #     batch_size=50,
 #     num_epochs=1,
 #     shuffle=False)
 # eval_results = audio_classifier.evaluate(input_fn=eval_input_fn)
 # print(eval_results)

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      batch_size=50,
      num_epochs=1,
      shuffle=False)
  eval_results = audio_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  predictions = list(audio_classifier.predict(input_fn=eval_input_fn))
  predicted_classes = [p["classes"] for p in predictions]
  with tf.Session() as sess:
      confusion_matrix = tf.confusion_matrix(labels=test_labels, predictions=predicted_classes, num_classes=6)
      confusion_matrix_to_Print = sess.run(confusion_matrix)
      print(confusion_matrix_to_Print)

if __name__ == "__main__":
  tf.app.run()
