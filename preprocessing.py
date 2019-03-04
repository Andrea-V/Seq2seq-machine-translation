import re
import tensorflow as tf
from language_index import LanguageIndex

# basic preprocessing
def preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r" \1 ", w) # metto lo spazio attorno ogni segno di punteggiatura
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,èéòàùì]+", " ", w) # elimino tutti i caratteri inusuali (tranne lettere accentate)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

def max_length(tensor):
    return max(len(t) for t in tensor)

def load_dataset(path, n_examples):
    # creo coppie di parole
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:n_examples]]
    
    #creo indici
    inp_lang = LanguageIndex(it for en, it in pairs)
    targ_lang = LanguageIndex(en for en, it in pairs)
    
    # sostituisco parola con indice
    input = [[inp_lang.word2idx[s] for s in it.split(' ')] for en, it in pairs] # frasi italiane
    target = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, it in pairs]  # frasi inglesi
    
    # padding
    maxlen_input, maxlen_target = max_length(input), max_length(target)
    input = tf.keras.preprocessing.sequence.pad_sequences(input, maxlen=maxlen_input, padding='post')
    target = tf.keras.preprocessing.sequence.pad_sequences(target, maxlen=maxlen_target, padding='post')
    
    return input, target, inp_lang, targ_lang, maxlen_input, maxlen_target
