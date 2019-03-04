import tensorflow as tf

class Decoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, n_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.CuDNNGRU(self.n_units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
        # projection layer
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # layers per l'attention
        self.w1 = tf.keras.layers.Dense(self.n_units)
        self.w2 = tf.keras.layers.Dense(self.n_units)
        self.v = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # aggiusto dimensione (time axis) per calcolare score
        hidden_t = tf.expand_dims(hidden, 1)
        
        # calcolo context vector
        score = tf.nn.tanh(self.w1(enc_output) + self.w2(hidden_t))
        attention_weights = tf.nn.softmax(self.v(score), axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) # includo context vector nell'input
        
        output, state = self.gru(x)

        # restituisco lista delle predizioni (più comoda per calcolare loss)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights
