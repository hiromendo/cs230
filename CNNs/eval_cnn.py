from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from cnn import cnnModel

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  # Load training and eval data
  
  test_data = np.load("datasets/testX1K.npy")
  test_labels = np.asarray(np.load("datasets/testY1K.npy"),dtype=np.int32)

  # Create the Estimator
  audio_classifier = tf.estimator.Estimator(model_fn=cnnModel, model_dir="checkpoint_path")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Evaluate the model and print results
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
  test_results = audio_classifier.evaluate(input_fn=test_input_fn)
  print(test_results)

if __name__ == "__main__":
  tf.app.run()
  

