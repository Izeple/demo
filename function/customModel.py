class CNN_Encoder_LSTM(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder_LSTM, self).__init__()
        # shape after dn == (batch_size, 100, embedding_dim)
        self.dn1 = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.dn1(x)
        x = tf.nn.relu(x)
        return x

class CNN_Encoder_GRU(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder_GRU, self).__init__()
        # shape after dn == (batch_size, 100, embedding_dim)
        self.dn1 = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.dn1(x)
        x = tf.nn.relu(x)
        return x

class Attention(tf.keras.Model):
  def __init__(self, units):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # CNN encoder output.shape == (batch_size, 100, embedding_dim)

    # change dimension for computing attention score
    hidden_from_last_output = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 100, units)  
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_from_last_output)))

    #compute score with 1 hidden layer
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 100, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class GRU_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(GRU_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.gru1 = tf.keras.layers.GRU(self.units,return_sequences=True,
                                   return_state=True)
    self.gru2 = tf.keras.layers.GRU(self.units,return_state=True)
    self.dn1 = tf.keras.layers.Dense(self.units)
    self.dn2 = tf.keras.layers.Dense(vocab_size)

    self.attention = Attention(self.units)

  def call(self, x, features, hidden):
    # defining attention 
    context_vector, attention_weights = self.attention(features, hidden)

    x = self.embedding(x)
    
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    x = self.gru(x)
    x = self.gru1(x)
    output, state = self.gru2(x)
    x = self.dn1(output)
    x = self.dn2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


class LSTM_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(LSTM_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.ls1 = tf.keras.layers.LSTM(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.ls2 = tf.keras.layers.LSTM(self.units,return_sequences=True,
                                   return_state=True)
    self.ls3 = tf.keras.layers.LSTM(self.units,return_state=True,return_sequences=True)
    self.dn1 = tf.keras.layers.Dense(self.units)
    self.dn2 = tf.keras.layers.Dense(vocab_size)

    self.attention = Attention(self.units)

  def call(self, x, features, hidden):
    # defining attention 
    context_vector, attention_weights = self.attention(features, hidden)

    x = self.embedding(x)

    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    x = self.ls1(x)
    x = self.ls2(x)
    output, state, statec = self.ls3(x)

    x = self.dn1(output)
    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.dn2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))