from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def confusion_matrix_op(logits, labels, num_classes):
    with tf.variable_scope('confusion_matrix'):
        # handle fully convolutional classifiers
        logits_shape = logits.shape
        if len(logits_shape) == 4 and logits_shape[1:3] == [1, 1]:
            top_k_logits = tf.squeeze(logits, [1, 2])
        else:
            top_k_logits = logits

        #Extract the predicted label (top-1)
        _, top_predicted_label = tf.nn.top_k(top_k_logits, k=1, sorted=False)
        # (batch_size, k) -> k = 1 -> (batch_size)
        top_predicted_label = tf.squeeze(top_predicted_label, axis=1)

        return tf.confusion_matrix(labels, top_predicted_label, num_classes=num_classes) 


def cnn_model_fn(features, labels, mode):

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 300, 300, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Input Tensor Shape: [batch_size, 300, 300, 1]
  # Output Tensor Shape: [batch_size, 148, 148, 32]
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5, 5],strides=(2,2),padding="valid",activation=tf.nn.relu)
  
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 148, 148, 32]
  # Output Tensor Shape: [batch_size, 72, 72, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[5, 5], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 72, 72, 32]
  # Output Tensor Shape: [batch_size, 34, 34, 64]
  conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],strides=(2,2),padding="valid",activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 34, 34, 64]
  # Output Tensor Shape: [batch_size, 17, 17, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 17, 17, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 17 * 17 * 64])

  # Dense Layer # 1
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.8 probability that element will be kept
  dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer # 2 (Output)
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, c] #NEEDS TO BE UPDATED WITH APPROPRIATE # OF CLASSES #6
  logits = tf.layers.dense(inputs=dropout, units=6)

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
 # init_op = tf.initialize_all_variables()
 # cf_matrix = tf.confusion_matrix(labels,predictions["classes"])
 # sess = tf.Session()
 # with sess.as_default():
  #    print(sess.run(cf_matrix)
  cm = confusion_matrix_op(logits, labels, 6)
  #sess = tf.Session()
  
  #with sess.as_default():
  #sess = tf.InteractiveSession()
  print(cm)
  #print(cm.eval())
  #sess.close()
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval dat
  train_data = np.load("dataset/x_train105K_6lang.npy")
  train_labels = np.asarray(np.load("dataset/y_train105K_6lang.npy"),dtype=np.int32)
  eval_data = np.load("dataset/x_val22_5K_6lang.npy")
  eval_labels = np.asarray(np.load("dataset/y_val22_5K_6lang.npy"),dtype=np.int32)
  test_data = np.load("dataset/x_test22_5K_6lang.npy")
  test_labels = np.load("dataset/y_test22_5K_6lang.npy")

  # Create the Estimator
  audio_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="basic_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor ion_matrixith label "probabilities"
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #logging_hook = tf.train.LoggingTensorHook(
      #tensors=tensors_to_log, every_n_iter=50)

  # Train the mode
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  audio_classifier.train(
      input_fn=train_input_fn,
      steps=1)
      #hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      #x={"x": train_data},
      #y=train_labels,
      x={"x": eval_data},
      y=eval_labels,
      #x={"x": test_data},
      #y=test_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = audio_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  predictions = list(audio_classifier.predict(input_fn=eval_input_fn))
  predicted_classes = [p["classes"] for p in predictions]
  with tf.Session() as sess:
      confusion_matrix = tf.confusion_matrix(labels=eval_labels, predictions=predicted_classes, num_classes=6)
      confusion_matrix_to_Print = sess.run(confusion_matrix)
      print(confusion_matrix_to_Print)

if __name__ == "__main__":
  tf.app.run()
