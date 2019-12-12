import sys
import numpy as np
import tensorflow as tf
import keras.layers as L


class Model:
    def __init__(self, name, inp_voc, out_voc,
                 emb_size=512, hid_size=512, attn_size=512):
        """ Translation model that uses attention. See instructions above. TODO tips"""
        self.name = name
        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.emb_size, self.hid_size = emb_size, hid_size

        with tf.variable_scope(name):
            self.emb_inp = L.Embedding(len(inp_voc), emb_size, name='embedding_1')
            self.emb_out = L.Embedding(len(out_voc), emb_size, name='embedding_2')
            self.enc_fw = LSTMCell('enc_fw', emb_size, hid_size)
            self.enc_bw = LSTMCell('enc_bw', emb_size, hid_size)

            self.dec = LSTMCell('decoder', emb_size + hid_size * 2, hid_size)
            self.attn = AttentionLayer('attn', 2 * hid_size, hid_size, attn_size)
            self.logits = L.Dense(len(out_voc), name='dense_1')

            # prepare to translate_lines
            self.inp = tf.placeholder('int32', [None, None])
            self.initial_state = self.encode(self.inp)
            self.prev_state = [tf.placeholder(x.dtype, x.shape) for x in self.initial_state]
            self.prev_tokens = tf.placeholder('int32', [None])
            self.next_state, self.next_logits = self.decode(self.prev_state, self.prev_tokens)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """

        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        enc_mask = infer_mask(inp, self.inp_voc.eos_ix)
        inp_emb = self.emb_inp(inp)

        enc_seq, _ = tf.nn.bidirectional_dynamic_rnn(
            self.enc_fw,
            self.enc_bw,
            inp_emb,
            sequence_length=inp_lengths,
            time_major=False,
            dtype=inp_emb.dtype
        ) # enc_seq: 2 x [batch, time, units]
        enc_seq = tf.concat(enc_seq, 2) #concat forward and backward rnn

        # initial decoder state
        batch_size = tf.shape(inp_emb)[0]
        hid0, cell0 = [tf.zeros([batch_size, self.hid_size])] * 2

        # attention
        attn0, attn_probas0 = self.attn(enc_seq, hid0, enc_mask)
        return [hid0, cell0, enc_seq, enc_mask, attn0, attn_probas0]

    def decode(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """
        [prev_hid, prev_cell, enc_seq, enc_mask, prev_attn, prev_probas_prev] = prev_state
        out_emb = self.emb_out(prev_tokens)
        inputs = tf.concat([out_emb, prev_attn], 1)
        dec_out, (new_hid, new_cell) = self.dec(inputs, (prev_hid, prev_cell))
        attn, attn_probas_new = self.attn(enc_seq, new_hid, enc_mask)
        output_logits = self.logits(tf.concat([out_emb, dec_out, attn], 1))

        return [new_hid, new_cell, enc_seq, enc_mask, attn, attn_probas_new], output_logits

    def translate_lines(self, inp_lines, max_len=100):
        """
        Translates a list of lines by greedily selecting most likely next token at each step
        :returns: a list of output lines, a sequence of model states at each step
        """
        sess = tf.get_default_session()
        state = sess.run(self.initial_state, {self.inp: self.inp_voc.to_matrix(inp_lines)})
        outputs = [[self.out_voc.bos_ix] for _ in range(len(inp_lines))]
        all_states = [state]
        finished = [False] * len(inp_lines)

        for t in range(max_len):
            state, logits = sess.run([self.next_state, self.next_logits],
                                     {**dict(zip(self.prev_state, state)),
                                      self.prev_tokens: [out_i[-1] for out_i in outputs]})
            next_tokens = np.argmax(logits, axis=-1)
            all_states.append(state)
            for i in range(len(next_tokens)):
                outputs[i].append(next_tokens[i])
                finished[i] |= next_tokens[i] == self.out_voc.eos_ix
        return self.out_voc.to_lines(outputs), all_states



class AttentionLayer:
    def __init__(self, name, enc_size, dec_size, hid_size, activ=tf.tanh,):
        """ A layer that computes additive attention response and weights """
        self.name = name
        self.enc_size = enc_size # num units in encoder state
        self.dec_size = dec_size # num units in decoder state
        self.hid_size = hid_size # attention layer hidden units
        self.activ = activ       # attention layer hidden nonlinearity

        with tf.variable_scope(name):
            self.dec = tf.get_variable('dec', shape=[dec_size, hid_size])
            self.enc = tf.get_variable('enc', shape=[enc_size, hid_size])
            self.vec = tf.get_variable('vec', shape=[hid_size, 1])[:, 0]

    def __call__(self, enc, dec, inp_mask):
        """
        Computes attention response and weights
        :param enc: encoder activation sequence, float32[batch_size, ninp, enc_size]
        :param dec: single decoder state used as "query", float32[batch_size, dec_size]
        :param inp_mask: mask on enc activatons (0 after first eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - attention response vector (weighted sum of enc)
            - probs - attention weights after softmax
        """
        with tf.variable_scope(self.name):
            hid = self.activ( # [batch_size, ninp, hid_size]
                tf.einsum('bte,eh->bth', enc, self.enc) +
                tf.expand_dims(tf.matmul(dec, self.dec), 1)
                )
            scores = tf.einsum('bth,h->bt', hid, self.vec) # [batch_size, ninp]

            # Compute scores.
            scores -= tf.reduce_max(scores, axis=1, keepdims=True)
            scores -= (1 - inp_mask) * 1000
            # Compute probabilities
            scores_exp = tf.exp(scores)
            Z = tf.reduce_sum(scores_exp, axis=1, keepdims=True)
            probs = scores_exp / Z

            # Compose attention.
            attn = tf.reduce_sum(probs[..., None] * enc, axis=1)
            
            return attn, probs


class LSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    An minimalistic implementation of LSTM cell that doesn't
     cause name conflict if several lstms are created in the same scope
    """

    def __init__(
            self, name, inp_size, hid_size,
            activ_f=tf.sigmoid, activ_i=tf.sigmoid, activ_o=tf.sigmoid,
            activ_h=tf.tanh, forget_bias=0, recurrent_dropout=0,
    ):
        self._name = name
        self.hid_size = hid_size
        self.activ_f = activ_f
        self.activ_i = activ_i
        self.activ_o = activ_o
        self.activ_h = activ_h
        self.forget_bias = forget_bias
        self.recurrent_dropout = recurrent_dropout

        with tf.variable_scope(name):

            def create_gate(name):
                i = tf.get_variable('i2' + name, shape=[inp_size, hid_size])
                h = tf.get_variable('h2' + name, shape=[self.hid_size, self.hid_size])
                b = tf.get_variable('b_' + name, shape=[self.hid_size])
                return (i, h, b)

            self.i2c, self.h2c, self.b_c = create_gate('c')
            self.i2f, self.h2f, self.b_f = create_gate('f')
            self.i2i, self.h2i, self.b_i = create_gate('i')
            self.i2o, self.h2o, self.b_o = create_gate('o')

    @property
    def state_size(self):
        return (self.hid_size, self.hid_size)

    @property
    def output_size(self):
        return self.hid_size

    def __call__(self, inputs, state, scope=None, enable_dropout=False):
        """
        inputs: [batch_size, inp_size]
        state: ([batch_size, hid_size], [batch_size, hid_size])
        -------------------------------
        outputs: [batch_size, hid_size]
        """
        h_state, c_state = state

        with tf.variable_scope(self._name):
            c = self.activ_h(tf.matmul(inputs, self.i2c) +
                             tf.matmul(h_state, self.h2c) +
                             self.b_c)
            if enable_dropout:
                c = tf.nn.dropout(c, 1 - self.recurrent_dropout)
            f = tf.matmul(inputs, self.i2f) + tf.matmul(h_state, self.h2f) + self.b_f
            i = tf.matmul(inputs, self.i2i) + tf.matmul(h_state, self.h2i) + self.b_i
            c = self.activ_f(f + self.forget_bias) * c_state + self.activ_i(i) * c
            o = tf.matmul(inputs, self.i2o) + tf.matmul(h_state, self.h2o) + self.b_o
            h = self.activ_o(o) * self.activ_h(c)
            return h, (h, c)


class Vocab:
    def __init__(self, tokens, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        """
        A special class that converts lines of tokens into matrices and backwards
        """
        assert all(tok in tokens for tok in (bos, eos, unk))
        self.tokens = tokens
        self.token_to_ix = {t: i for i, t in enumerate(tokens)}
        self.bos, self.eos, self.unk = bos, eos, unk
        self.bos_ix = self.token_to_ix[bos]
        self.eos_ix = self.token_to_ix[eos]
        self.unk_ix = self.token_to_ix[unk]

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_lines(lines, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        flat_lines = '\n'.join(list(lines)).split()
        tokens = sorted(set(flat_lines))
        tokens = [t for t in tokens if t not in (bos, eos, unk) and len(t)]
        tokens = [bos, eos, unk] + tokens
        return Vocab(tokens, bos, eos, unk)

    def tokenize(self, string):
        """converts string to a list of tokens"""
        tokens = [tok if tok in self.token_to_ix else self.unk
                  for tok in string.split()]
        return [self.bos] + tokens + [self.eos]

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

        matrix = np.zeros((len(lines), max_len), dtype='int32') + self.eos_ix
        for i, seq in enumerate(lines):
            row_ix = list(map(self.token_to_ix.get, seq))[:max_len]
            matrix[i, :len(row_ix)] = row_ix

        return matrix

    def to_lines(self, matrix, crop=True):
        """
        Convert matrix of token ids into strings
        :param matrix: matrix of tokens of int32, shape=[batch,time]
        :param crop: if True, crops BOS and EOS from line
        :return:
        """
        lines = []
        for line_ix in map(list, matrix):
            if crop:
                if line_ix[0] == self.bos_ix:
                    line_ix = line_ix[1:]
                if self.eos_ix in line_ix:
                    line_ix = line_ix[:line_ix.index(self.eos_ix)]
            line = ' '.join(self.tokens[i] for i in line_ix)
            lines.append(line)
        return lines


### Utility TF functions ###


def infer_length(seq, eos_ix, time_major=False, dtype=tf.int32):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: lengths, int32 vector of shape [batch]
    """
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos_ix), dtype)
    count_eos = tf.cumsum(is_eos, axis=axis, exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos, 0), dtype), axis=axis)
    return lengths


def infer_mask(seq, eos_ix, time_major=False, dtype=tf.float32):
    """
    compute mask given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    axis = 0 if time_major else 1
    lengths = infer_length(seq, eos_ix, time_major=time_major)
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
    batch_i = tf.tile(tf.range(0, batch_size)[:, None], [1, seq_len])
    time_i = tf.tile(tf.range(0, seq_len)[None, :], [batch_size, 1])
    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)

    return tf.gather_nd(values, indices_nd)


def save(variables, path, sess=None):
    """
    saves variable weights independently (without tf graph)
    :param variables: an iterable of TF variables
    """
    sess = sess or tf.get_default_session()
    assert sess is not None, "please make sure you defined a default TF session"
    var_values = sess.run({w.name: w for w in variables})
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
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
