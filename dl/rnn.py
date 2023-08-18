'''

'''

from _imports import *


# Recurrent Neural Network class
class TextRNN(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim, rnn_units, _c2i, _i2c):
    super().__init__(self)
    # layers
    self.embedding = tf.keras.layers.Embedding(
      input_dim=vocab_size,
      output_dim=embedding_dim
    )
    self.lstm = tf.keras.layers.LSTM(
      units=rnn_units,
      return_sequences=True
    )
    self.dense = tf.keras.layers.Dense(
      units=vocab_size
    )

    # mappings
    self.c2i = _c2i
    self.i2c = _i2c


  def call(self, inputs):
    ret = self.embedding(inputs)
    ret = self.lstm(ret)
    return self.dense(ret)

  # vectorize text to nums
  def text_to_ids(self, txt):
    return [self.c2i[c] for c in txt]

  # turn nums to txt
  def ids_to_text(self, nums):
    return [self.i2c[i] for i in nums]


# generate sequences for training
def gen_seqs(vectorized_text, batch_size, seq_length):
  vec_length = len(vectorized_text)
  ids = np.random.randint(low=0, high=vec_length-seq_length, size=batch_size)
  xs, ys = np.array([vectorized_text[a:a+seq_length] for a in ids]), np.array([vectorized_text[b+1:b+1+seq_length] for b in ids])
  return xs, ys

# categorical cross entropy loss
def catXent_loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# main
if __name__ == '__main__':
 
  tf.random.set_seed(33)

  # sample data from tutorial: https://www.tensorflow.org/text/tutorials/text_generation
  skspr = open('<text file path>', 'rb').read().decode(encoding='utf-8')
  print(f'# of characters in file: {len(skspr)}')
  
  # unique chars in text
  vocab = set(skspr)
  print(f'# of unique characters: {len(vocab)}')

  # vectorize text
  chr2num = {c:i for i, c in enumerate(vocab)}
  num2chr = list(vocab)
  
  # model
  batch_size = 8
  seq_length = 100
  embedding_dim = 256
  rnn_units = 512
  vocab_size = len(vocab)
  myRNN = TextRNN(vocab_size, embedding_dim, rnn_units, chr2num, num2chr)
  
  # prediction on untrained model
  vec_skspr = myRNN.text_to_ids(skspr)
  xs, ys = gen_seqs(vec_skspr, batch_size, seq_length)
  eg_preds = myRNN(xs)
  myRNN.summary()
  eg_pred = tf.random.categorical(logits=eg_preds[0], num_samples=1) # sampling from probability distribution
  eg_pred = tf.squeeze(eg_pred, axis=-1).numpy()
  print(f"Prediction example from untrained model:\n"
    f"Input:\n{''.join(myRNN.ids_to_text(xs[0]))}\n\n"
    f"Predictions:\n{''.join(myRNN.ids_to_text(eg_pred))}\n\n"
    f"Exp. of mean Cat. cross entropy loss (close to vocab size indicates predictions are equal across all categories initially):\n"
    f"{np.exp(catXent_loss(ys, eg_preds).numpy().mean())}\n"
    f"-------------------------------------------"
  )

  # training
  epochs = 50
  learning_rate = 8e-3
  myRNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=catXent_loss)
  history = myRNN.fit(x=xs, y=ys, epochs=epochs)
  
  # loss visualization
  losses = history.history['loss']
  sns.lineplot(x=range(1, len(losses)+1), y=losses, color='hotpink')
  plt.title(label='Mean cat. cross entropy error loss progression with training')
  plt.xlabel(xlabel='Epoch')
  plt.ylabel(ylabel='Loss')
  plt.tight_layout()
  plt.show()


  # generating new text from single character input

  input_chars = "Q"
  gen_size = 200
  gen_text = []
  # Convert strings to token IDs.
  input_ids = np.array([myRNN.text_to_ids(input_chars)]).reshape(1, -1)

  for _ in range(gen_size):
    # Run the model: predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits = myRNN(input_ids)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)
    # Convert from token ids to characters
    predicted_chars = myRNN.ids_to_text(tf.reshape(predicted_ids, [-1]).numpy())
    gen_text.append(predicted_chars[0])
    # feed back predicted id
    input_ids = predicted_ids.numpy().reshape(1, -1)

  print(f"Generated text from '{input_chars}':\n"
    f"{''.join(gen_text)}\n"      
  )
