# build & train a single-layer neural network with keras

from _imports import *


# sub class Model representing single-layer NN
class SingleLayerNN(tf.keras.Model):
    
  def __init__(self, n_outs):
    super(SingleLayerNN, self).__init__()

    # layer
    self.layer1 = tf.keras.layers.Dense(units=n_outs, activation=tf.nn.sigmoid)

    # optimizer algo, loss function, accuracy measure
    self.compile(
      optimizer=tf.keras.optimizers.experimental.SGD(), # stochastic gradient descent
      loss=tf.keras.losses.mean_squared_error, # loss function
      metrics=tf.keras.metrics.mean_squared_error,
    )


  def train(self, x_train, y_train):
    return self.fit(
      x=x_train,
      y=y_train,
      batch_size=10,
      epochs=100 # iterations
    )


  # get output of NN
  def call(self, inputs):
    return self.layer1(inputs)


# main
if __name__ == '__main__':

  mySLNN = SingleLayerNN(2)

  xs = np.random.rand(100, 3)
  ys = np.random.rand(100, 2)

  # capture params for each training step
  history = mySLNN.train(xs, ys)
  
  print(f'Final weights: {mySLNN.get_weights()}')

  # example input
  x = tf.constant([[2.0, 0.5, 1.2]], dtype=tf.float64)
  print(f'For input {x}, the final NN gives {mySLNN.call(x)}')

  losses = history.history['loss']
  sns.lineplot(x=range(1, len(losses)+1), y=losses, color='hotpink')
  plt.title(label='Mean squared error loss of NN with each training step')
  plt.xlabel(xlabel='Epoch')
  plt.ylabel(ylabel='Loss')
  plt.tight_layout()
  plt.show()
