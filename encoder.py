import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, n_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.bigru = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(self.n_units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform'))
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_fwd, state_bwd = self.bigru(x, initial_state = [hidden, hidden])
        
        # merging forward and backward state
        state = tf.keras.layers.Concatenate(axis=1)([state_fwd, state_bwd])
        
        return output, state
