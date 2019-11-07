"""
vocabulary and utility functions similar to the ones you used in week4, but this time there's no BOS/EOS tokens
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
tokenizer = WordPunctTokenizer()
tokenize = lambda sent: ' '.join(tokenizer.tokenize(sent.lower()))

def build_dataset(path, tokenized=False, verbose=False, min_chars_sent=10,
                  random_state=42, test_size=0.25,
                  prune_impossible=True,  prune_all_correct=True):
    '''
    :param path: path to SQuAD data, e.g. https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
    :param tokenized: if True, data is lowercased and processed with WordPunctTokenizer
    :param verbose: if True, displays progressbar
    :returns: [train, test] - each is a dataframe with (question, possible_answers, correct_answer_indices) as columns
    '''
    data = json.load(open(path))['data']
    dataset = []
    for row in (tqdm(data) if verbose else data):
        for paragraph_id, paragraph in enumerate(row['paragraphs']):
            current_sent, current_index = '', 0
            sentences, offsets = [], []
            for raw_sent in sent_tokenize(paragraph['context']):
                current_sent += ' ' + raw_sent
                if len(current_sent) > min_chars_sent:
                    current_index += len(current_sent)
                    sentences.append(current_sent.strip())
                    offsets.append(current_index)
                    current_sent = ''
                    
            if len(current_sent):
                sentences.append(current_sent.strip())
                offsets.append(len(paragraph['context']))
                current_sent = ''
            
            if tokenized:
                sentences_tok = list(map(tokenize, sentences))
            
            for qa in paragraph['qas']:
                question = qa['question']
                correct_indices = set()
                for answer in qa['answers']:
                    #find a sentence that contains an answer
                    for i, offset, sent in zip(range(len(offsets)), offsets, sentences):
                        if answer['answer_start'] < offset:
                            correct_indices.add(i)
                            break
                    else:
                        raise ValueError('error: correct answer not found')
                
                if tokenized:
                    question_tok = tokenize(question)
                if prune_impossible and not len(correct_indices): 
                    continue
                if prune_all_correct and len(sentences) <= len(correct_indices):
                    continue
                    
                wrong_indices = [i for i in range(len(sentences)) if i not in correct_indices]
                dataset.append((paragraph_id,
                                question_tok if tokenized else question,
                                sentences_tok if tokenized else sentences, 
                                sorted(correct_indices), sorted(wrong_indices)))
                
    data = pd.DataFrame(dataset, columns=['paragraph_id', 'question', 'options', 'correct_indices', 'wrong_indices'])
    train_pid, test_pid = map(set, train_test_split(data['paragraph_id'].unique(), test_size=test_size,
                                                    random_state=random_state))
    is_train = data['paragraph_id'].apply(lambda pid: pid in train_pid)
    return data.loc[is_train], data.loc[~is_train]


class Vocab:
    def __init__(self, tokens, pad="_PAD_", unk='_UNK_'):
        """
        A special class that converts lines of tokens into matrices and backwards
        """
        assert all(tok in tokens for tok in (pad, unk))
        self.tokens = tokens
        self.token_to_ix = {t:i for i, t in enumerate(tokens)}
        self.pad, self.unk = pad, unk
        self.pad_ix = self.token_to_ix.get(pad)
        self.unk_ix = self.token_to_ix.get(unk)

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_lines(lines, pad="_PAD_", unk='_UNK_'):
        flat_lines = '\n'.join(list(lines)).split()
        tokens = sorted(set(flat_lines))
        tokens = [t for t in tokens if t not in (pad, unk) and len(t)]
        tokens = [pad, unk] + tokens
        return Vocab(tokens, pad, unk)

    def tokenize(self, string):
        """converts string to a list of tokens"""
        tokens = [tok if tok in self.token_to_ix else self.unk
                  for tok in string.split()]
        return tokens

    def to_matrix(self, lines, max_len=None):
        """
        convert variable length token sequences into  fixed size matrix
        example usage:
        >>>print( as_matrix(words[:3],source_to_ix))
        [[15 22 21 28 27 13 -1 -1 -1 -1 -1]
         [30 21 15 15 21 14 28 27 13 -1 -1]
         [25 37 31 34 21 20 37 21 28 19 13]]
        """
        lines = list(map(self.tokenize, lines))
        max_len = max_len or max(map(len, lines))
        matrix = np.full((len(lines), max_len), self.pad_ix, dtype='int32')
        for i, seq in enumerate(lines):
            row_ix = list(map(self.token_to_ix.get, seq))[:max_len]
            matrix[i, :len(row_ix)] = row_ix
        return matrix

    def to_lines(self, matrix, crop=True):
        """
        Convert matrix of token ids into strings
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        :param crop: if True, crops PAD from line
        :return:
        """
        lines = []
        for line_ix in map(list,matrix):
            if crop:
                if self.pad_ix in line_ix:
                    line_ix = line_ix[:line_ix.index(self.pad_ix)]
            line = ' '.join(self.tokens[i] for i in line_ix)
            lines.append(line)
        return lines


### Utility TF functions ###


def infer_length(seq, pad_ix=1, time_major=False, dtype=tf.int32):
    """
    compute length given output indices and PAD code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param pad_ix: integer index of end-of-sentence token
    :returns: lengths, int32 vector of shape [batch]
    """
    axis = 0 if time_major else 1
    is_pad = tf.cast(tf.equal(seq, pad_ix), dtype)
    count_pad = tf.cumsum(is_pad, axis=axis, exclusive=False)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_pad, 0), dtype), axis=axis)
    return lengths


def infer_mask(seq, pad_ix=1, time_major=False, dtype=tf.float32):
    """
    compute mask given output indices and PAD code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param pad_ix: integer index of end-of-sentence token
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    axis = 0 if time_major else 1
    lengths = infer_length(seq, pad_ix, time_major=time_major)
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(seq)[axis], dtype=dtype)
    if time_major: mask = tf.transpose(mask)
    return mask


def select_values_over_last_axis(values, indices):
    """
    Auxiliary function to select logits corresponding to chosen tokens.
    :param values: logits for all actions: float32[batch,tick,action]
    :param indices: action ids int32[batch,tick]
    :returns: values selected for the given actions: float[batch,tick]
    """
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size, seq_len = tf.shape(indices)[0], tf.shape(indices)[1]
    batch_i = tf.tile(tf.range(0, batch_size)[:, None],[1, seq_len])
    time_i = tf.tile(tf.range(0, seq_len)[None, :], [batch_size, 1])
    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)

    return tf.gather_nd(values,indices_nd)


def save(variables, path, sess=None):
    """
    saves variable weights independently (without tf graph)
    :param variables: an iterable of TF variables
    """
    sess = sess or tf.get_default_session()
    assert sess is not None, "please make sure you defined a default TF session"
    var_values = sess.run({w.name : w for w in variables})
    np.savez(path, **var_values)


def load(variables, path, sess=None, verbose=True):
    """
    loads variable weights saved with save function above
    :param variables: a list/tuple of 
    """
    sess = sess or tf.get_default_session()
    assert sess is not None, "please make sure you defined a default TF session"
    var_values = np.load(path)
    var_values = {name: var_values[name] for name in var_values}
    not_initialized = []
    ops = []
    for var in variables:
        if var.name in var_values:
            ops.append(tf.assign(var, var_values.pop(var.name)))
        else:
            not_initialized.append(var.name)
    sess.run(ops)
    if verbose:
        if len(var_values):
            print('Checkpoint weights not used:', ' '.join(var_values.keys()), file=sys.stderr)
        if len(not_initialized):
            print('Variables not initialized:', ' '.join(not_initialized), file=sys.stderr)
    


def initialize_uninitialized(sess=None):
    """
    Initialize unitialized variables, doesn't affect those already initialized
    :param sess: in which session to initialize stuff. Defaults to tf.get_default_session()
    """
    sess = sess or tf.get_default_session()
    assert sess is not None, "please make sure you defined a default TF session"
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
