import numpy as np
import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing import *
from encoder import Encoder
from decoder import Decoder
from eval import *

import progressbar
from nltk.translate.bleu_score import sentence_bleu

tf.enable_eager_execution()
TRAIN = True
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



# load e preprocessing dataset
N_EXAMPLES = 250000 # 295358
path = "./ita-eng/ita.txt"
inputs, target, input_lang, target_lang, maxlen_input, maxlen_target = load_dataset(path, N_EXAMPLES)
input_train, input_val, target_train, target_val = train_test_split(inputs, target, test_size=0.2)

# costanti
BUFFER_SIZE = len(input_train)
BATCH_SIZE = 64
N_BATCH_TR = BUFFER_SIZE // BATCH_SIZE
N_BATCH_VAL = len(input_val) // BATCH_SIZE
EMBEDDING_SIZE = 256
N_UNITS = 1024
VOCAB_INPUT_SIZE = len(input_lang.word2idx)
VOCAB_TARGET_SIZE = len(target_lang.word2idx)

# creo tf.Dataset
dataset_tr = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(BUFFER_SIZE)
dataset_tr = dataset_tr.batch(BATCH_SIZE, drop_remainder=True)
dataset_val = tf.data.Dataset.from_tensor_slices((input_val, target_val)).shuffle(len(input_val))
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)

# creo encoder e decoder
encoder = Encoder(VOCAB_INPUT_SIZE, EMBEDDING_SIZE, N_UNITS // 2, BATCH_SIZE)
decoder = Decoder(VOCAB_TARGET_SIZE, EMBEDDING_SIZE, N_UNITS, BATCH_SIZE)

# optimizer e loss function
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

# imposto checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

## CICLO DI TRAINING ##
if TRAIN:
    N_EPOCHS = 10
    bar = progressbar.ProgressBar(max_value=N_EPOCHS)
    for epoch in range(N_EPOCHS):
        start = time.time()
        
        hidden = tf.zeros((BATCH_SIZE, N_UNITS // 2))
        total_tr_loss = 0
        total_val_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset_tr):
            loss = 0
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([target_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
                
                for t in range(1, targ.shape[1]):
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions)
                    # uso teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            total_tr_loss += batch_loss
            
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} of {} TR Loss {:.4f}'.format(epoch + 1, batch, N_BATCH_TR, 
                                                                            batch_loss.numpy()))
        # checkpoint dopo ogni epoca
        checkpoint.save(file_prefix = checkpoint_prefix)
        
        # validation loss per controllare l'overfitting
        for (batch, (inp, targ)) in enumerate(dataset_val):
            val_loss = 0
            enc_output, enc_hidden = encoder(inp, hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([target_lang.word2idx['<start>']] * BATCH_SIZE, 1)       

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                val_loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

            batch_val_loss = (val_loss / int(targ.shape[1]))
            total_val_loss += batch_val_loss    
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} of {} VL Loss {:.4f}'.format(epoch + 1, batch, N_BATCH_VAL, 
                                                                            batch_val_loss.numpy()))
    
        print('Epoch {} TR Loss {:.4f} VL Loss {:.4f}'.format(epoch + 1,
                                            total_tr_loss / N_BATCH_TR,
                                            total_val_loss / N_BATCH_VAL))
        print('Time taken for 1 epoch {:.1f} sec\n'.format(time.time() - start))
        bar.update(epoch+1)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# computing BLEU score on validation set
print("Computing BLEU scores...")
bleu_score = []
bleu_1_score = []
bleu_2_score = []
bleu_3_score = []
bleu_4_score = []
bar = progressbar.ProgressBar(max_value=N_BATCH_VAL)
bar.update(0)
for (batch, (inp, targ)) in enumerate(dataset_val):
    #print("Batch", batch, "of", N_BATCH_VAL)
    #batch_len  = targ.shape[0]
    for (sample, (input_sentence, target_sentence)) in enumerate(zip(inp, targ)):
        #print("Sample", sample, "of", batch_len)

        input_sentence  = [ input_lang.idx2word[idx] for idx in input_sentence.numpy()  ]
        target_sentence = [ target_lang.idx2word[idx] for idx in target_sentence.numpy() ] 

        input_sentence = list(filter(lambda x: x != "<start>" and x != "<end>" and x != "<pad>", input_sentence))
        target_sentence = list(filter(lambda x: x != "<start>" and x != "<end>" and x != "<pad>", target_sentence))
        input_sentence = " ".join(input_sentence)
        
        result, sentence, attention_plot = predict(input_sentence, encoder, decoder, input_lang, target_lang, maxlen_input, maxlen_target)
        
        candidate = list(filter(lambda x: x != "<end>",result.strip().split(" ")))
        reference = [target_sentence]

        bleu_score.append(sentence_bleu(reference, candidate))
        bleu_1_score.append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        bleu_2_score.append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
        bleu_3_score.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
        bleu_4_score.append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
    
    bar.update(batch+1)

print("BLUE Score:", np.mean(bleu_score))
print('Cumulative 1-gram:',  np.mean(bleu_1_score))
print('Cumulative 2-gram: ', np.mean(bleu_2_score))
print('Cumulative 3-gram: ', np.mean(bleu_3_score))
print('Cumulative 4-gram: ', np.mean(bleu_4_score))


translate('Prova a tradurre questa frase.', encoder, decoder, input_lang, target_lang, maxlen_input, maxlen_target)
