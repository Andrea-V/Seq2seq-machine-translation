import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from preprocessing import preprocess_sentence

def predict(sentence, encoder, decoder, input_lang, target_lang, maxlen_input, maxlen_target):
    attention_plot = np.zeros((maxlen_target, maxlen_input))
    
    sentence = preprocess_sentence(sentence)

    inputs = [input_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=maxlen_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = tf.zeros((1, encoder.n_units))
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang.word2idx['<start>']], 0)

    for t in range(maxlen_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # salvo attention weight per plot
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.multinomial(tf.exp(predictions), num_samples=1)[0][0].numpy()
        result += target_lang.idx2word[predicted_id] + ' '

        if target_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='gray')    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()

def translate(sentence, encoder, decoder, input_lang, target_lang, maxlen_input, maxlen_target):
    result, sentence, attention_plot = predict(sentence, encoder, decoder, input_lang, target_lang, maxlen_input, maxlen_target)
        
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))
    
    attention_plot = attention_plot[:len(result.split(' '))-1, :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))
