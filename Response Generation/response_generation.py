import os
import re
import pickle
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense 
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import gensim
from gensim import models
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.porter import *
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']


pwd = os.popen("pwd").read()[:-1]
path_to_respgen = f"{pwd}/Response Generation"


# Text Pre-processing

def _should_skip(line, min_length, max_length):
    """Whether a line should be skipped depending on the length."""
    return len(line) < min_length or len(line) > max_length


def create_example(previous_lines, line, file_id):
    """Creates examples with multi-line context
    The examples will include:
        file_id: the name of the file where these lines were obtained.
        response: the current line text
        context: the previous line text
        context/0: 2 lines before
        context/1: 3 lines before, etc.
    """
    example = {
        'file_id': file_id,
        'context': previous_lines[-1],
        'response': line,
    }
    example['file_id'] = file_id
    example['context'] = previous_lines[-1]

    extra_contexts = previous_lines[:-1]
    example.update({
        'context/{}'.format(i): context
        for i, context in enumerate(extra_contexts[::-1])
    })

    return example


def _preprocess_line(line):
    line = line.decode("utf-8")

    # Remove the first word if it is followed by colon (speaker names)
    # NOTE: this wont work if the speaker's name has more than one word
    line = re.sub('(?:^|(?:[.!?]\\s))(\\w+):', "", line)

    # Remove anything between brackets (corresponds to acoustic events).
    line = re.sub("[\\[(](.*?)[\\])]", "", line)

    # Strip blanks hyphens and line breaks
    line = line.strip(" -\n")

    return line


def _create_examples_from_file(file_name, min_length=0, max_length=20,
                               num_extra_contexts=5):

    previous_lines = []
    with open(file_name, 'rb') as f:
      for line in f :
        line = _preprocess_line(line)
        if not line:
            continue

        should_skip = _should_skip(
            line,
            min_length=min_length,
            max_length=max_length)

        if previous_lines:
            should_skip |= _should_skip(
                previous_lines[-1],
                min_length=min_length,
                max_length=max_length)

            if not should_skip:
                yield create_example(previous_lines, line, file_name)

        previous_lines.append(line.lower())
        if len(previous_lines) > num_extra_contexts + 1:
            del previous_lines[0]

in_comma = "'"

def remove_char(sentence):
  sent = sentence.replace('!', '')
  sent = sent.replace(',', '')
  sent = sent.replace(in_comma, '')
  sent = sent.replace('%', '')
  sent = sent.replace('-', '')
  sent = sent.replace('.', '')
  sent = sent.replace('?', '')
  sent = sent.replace('/', '')
  sent = sent.replace(':', '')
  sent = sent.replace(';', '')

  return sent

# Tokenizer

oov_token = "<OOV>"
max_length = 20
num_topic_words = 4

tokenizer = Tokenizer(oov_token=oov_token)

with open(f"{path_to_respgen}/bin/Tokens.txt", 'r') as file:
    js_string = file.read()
    tokenizer = tokenizer_from_json(js_string)
word_index = tokenizer.word_index
word_index['startsent'] = 0
word_index['endsent'] = len(word_index)+1
index_word = {word_index[word]:word for word in word_index}
vocab_size = len(word_index) + 1

def preprocess_sent(text_list):
    inputs = []
    for sent in text_list:
        inputs.append(remove_char(sent))
    input_seq = tokenizer.texts_to_sequences(inputs)    
    input_seq_pad = pad_sequences(input_seq, maxlen = max_length ,padding = 'post', truncating = 'post')
    return input_seq_pad 


# Word Embeddings

embeddings_index = {} 
f = open(f'{path_to_respgen}/bin/glove.42B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


embedding_dim = 300

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))            
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector



# Loading LDA Model
lda_model = models.ldamodel.LdaModel.load(f'{path_to_respgen}/bin/LDA/LDA.model', mmap = 'r')

# Preprocessing corpus for topic Modeling

stemmer = PorterStemmer()

def lemmatize_stemming(text):
    x = WordNetLemmatizer().lemmatize(text, pos='v')
    return stemmer.stem(x)
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
    return result


topic_dict = corpora.Dictionary.load(fname = f'{path_to_respgen}/bin/topic_dict.dict', mmap = 'r')
try :
    topic_dict.filter_extremes(no_below=10, no_above=0.5)
except:
    pass


def extract_topic(input):

  c = 0
  bow_vector = topic_dict.doc2bow(preprocess(input))
  for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
      wp = lda_model.show_topic(index)
      topic_keywords = ", ".join([word for word, prop in wp])
  return topic_keywords

def get_topic(input):

    input_topic = extract_topic(input)
    input_seq = tokenizer.texts_to_sequences([input_topic])
    input_seq_pad = pad_sequences(input_seq, maxlen = max_length ,padding = 'post', truncating = 'post')
    return input_seq_pad[0]

"""
    The Seq2Seq Topic Aware Model Definitions
"""



max_sequence_len = 40
batch_size = 400

class Encoder(tf.keras.Model):
    def __init__(self, hidden_size=1024, max_sequence_len=40, batch_size=batch_size, embedding_dim=300, vocab_size=vocab_size+1):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[embeddings_matrix], trainable=False)
        self.GRU_1 = GRU(units=hidden_size, return_sequences=True,recurrent_initializer='glorot_uniform')
        self.GRU_2 = GRU(units=hidden_size,
                         return_sequences=True, return_state=True,recurrent_initializer='glorot_uniform')

    def initial_hidden_state(self):
        return tf.zeros(shape=(self.batch_size, self.hidden_size))

    def call(self, x, initial_state, training=False):
        x = self.embedding_layer(x)
        x = self.GRU_1(x, initial_state=initial_state)
        x, hidden_state = self.GRU_2(x)
        return x, hidden_state


class Attention(tf.keras.Model):
    def __init__(self, hidden_size=256):
        super(Attention, self).__init__()
        self.fc1 = Dense(units=hidden_size)
        self.fc2 = Dense(units=hidden_size)
        self.fc3 = Dense(units=1)

    def call(self, encoder_output, hidden_state, training=False):
        '''hidden_state : h(t-1)'''
        y_hidden_state = tf.expand_dims(hidden_state, axis=1)
        y_hidden_state = self.fc1(y_hidden_state)
        y_enc_out = self.fc2(encoder_output)

        y = tf.keras.backend.tanh(y_enc_out + y_hidden_state)
        attention_score = self.fc3(y)
        attention_weights = tf.keras.backend.softmax(attention_score, axis=1)

        context_vector = tf.multiply(encoder_output, attention_weights)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, hidden_size=1024, max_sequence_len=40, batch_size=batch_size, embedding_dim=300, vocab_size=vocab_size+1):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
    
        self.embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[embeddings_matrix], trainable=False)
        self.GRU = GRU(units=hidden_size,
                       return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.attention = Attention(hidden_size=self.hidden_size)
        self.fc = Dense(units=self.vocab_size)

    def initial_hidden_state(self):
        return tf.zeros(shape=(self.batch_size, self.hidden_size))

    def call(self, x, encoder_output, hidden_state, training=False):
        x = self.embedding_layer(x)
        context_vector, attention_weights = self.attention(
            encoder_output, hidden_state, training=training)
        contect_vector = tf.expand_dims(context_vector, axis=1)
        x = tf.concat([x, contect_vector], axis=-1)
        x, curr_hidden_state = self.GRU(x)
        x = tf.reshape(x, shape=[self.batch_size, -1])
        x = self.fc(x)
        return x, curr_hidden_state, attention_weights

loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
train_accuracy = tf.metrics.SparseCategoricalAccuracy()

def loss_function(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    mask = 1 - tf.cast(tf.equal(y_true, 0), 'float32')
    return tf.reduce_mean(loss * mask)

encoder = Encoder()
decoder = Decoder()

checkpoint_dir = f'{path_to_respgen}/bin/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

try:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
except:
    print("no saved checkpoints found")

# Evaluation

def get_response(sentence):
    topic_sent = get_topic(sentence)
    sentence = preprocess_sent([sentence])
    new_sentence = np.concatenate((np.array(sentence[0]), np.array(topic_sent)))
    enc_init = tf.zeros(shape=[1, 1024])
    enc_out, enc_hidden = encoder(sentence, enc_init)
    decoder.batch_size = 1
    tokenizer.index_word[0] = ''
    decoded = []
    att = []
    current_word = tf.expand_dims([word_index['startsent']], axis=0) 
    decoder_hidden = enc_hidden
    for word_idx in range(1, max_sequence_len):
        logits, decoder_hidden, attention_weights = decoder(current_word, enc_out, decoder_hidden)
        decoded_idx = np.argmax(logits)
        if index_word[decoded_idx] == 'endsent':
            break
        decoded.append(tokenizer.index_word[decoded_idx])
        att.append(attention_weights.numpy().squeeze())
        current_word = tf.expand_dims([decoded_idx], axis=0)
    return ' '.join(decoded)
