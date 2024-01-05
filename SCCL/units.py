
import anndata
import h5py
import time
import os
import psutil
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from scipy import sparse
from SCCL.basenji_utils import *
from tensorflow import keras
from keras_multi_head import MultiHeadAttention
from tensorflow.keras import layers
###############################
# function for pre-processing #
###############################

OVER1 = 0.1
OVER2 = 0.9
def make_bed_seqs_from_df(input_bed, fasta_file, seq_len, stranded=False):
    """Return BED regions as sequences and regions as a list of coordinate
    tuples, extended to a specified length."""
    """Extract and extend BED sequences to seq_len."""
    fasta_open = pysam.Fastafile(fasta_file)

    seqs_dna = []
    seqs_coords = []

    for i in range(input_bed.shape[0]):
        chrm = input_bed.iloc[i,0]
        start = int(input_bed.iloc[i,1])
        end = int(input_bed.iloc[i,2])
        strand = "+"

        # determine sequence limits
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len

        # save
        if stranded:
            seqs_coords.append((chrm, seq_start, seq_end, strand))
        else:
            seqs_coords.append((chrm, seq_start, seq_end))
        # initialize sequence
        seq_dna = ""
        # add N's for left over reach
        if seq_start < 0:
            print(
                "Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),
                file=sys.stderr,
            )
            seq_dna = "N" * (-seq_start)
            seq_start = 0

        # get dna
        seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

        # add N's for right over reach
        if len(seq_dna) < seq_len:
            print(
                "Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),
                file=sys.stderr,
            )
            seq_dna += "N" * (seq_len - len(seq_dna))
        # append
        seqs_dna.append(seq_dna)
    fasta_open.close()
    return seqs_dna, seqs_coords


def dna_1hot_2vec(seq, seq_len=None):
    """dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
                 rather than sampling.
    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len, ), dtype="int8")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i] = 0
            elif nt == "C":
                seq_code[i] = 1
            elif nt == "G":
                seq_code[i] = 2
            elif nt == "T":
                seq_code[i] = 3
            else:
                seq_code[i] =  random.randint(0, 3)
    return seq_code

def split_train_test_val(ids, seed=10, train_ratio=0.9):
    np.random.seed(seed)
    test_val_ids = np.random.choice(
        ids,
        int(len(ids) * (1 - train_ratio)),
        replace=False,
    )
    train_ids = np.setdiff1d(ids, test_val_ids)
    val_ids = np.random.choice(
        test_val_ids,
        int(len(test_val_ids) / 2),
        replace=False,
    )
    test_ids = np.setdiff1d(test_val_ids, val_ids)
    return train_ids, test_ids, val_ids


def make_h5_sparse(tmp_ad, h5_name, input_fasta, seq_len=1344, batch_size=1000):
    ## batch_size: how many peaks to process at a time
    ## tmp_ad.var must have columns chr, start, end

    t0 = time.time()

    m = tmp_ad.X
    m = m.tocoo().transpose().tocsr()
    n_peaks = tmp_ad.shape[1]
    bed_df = tmp_ad.var.loc[:,['chr','start','end']] # bed file
    bed_df.index = np.arange(bed_df.shape[0])
    n_batch = int(np.floor(n_peaks/batch_size))
    batches = np.array_split(np.arange(n_peaks), n_batch) # split all peaks to process in batches

    ### create h5 file
    # X is a matrix of n_peaks * 1344
    f = h5py.File(h5_name, "w")

    ds_X = f.create_dataset(
        "X",
        (n_peaks, seq_len),
        dtype="int8",
    )

    # save to h5 file
    for i in range(len(batches)):

        idx = batches[i]
        # write X to h5 file
        seqs_dna,_ = make_bed_seqs_from_df(
            bed_df.iloc[idx,:],
            fasta_file=input_fasta,
            seq_len=seq_len,
        )
        dna_array_dense = [dna_1hot_2vec(x) for x in seqs_dna]
        dna_array_dense = np.array(dna_array_dense)
        ds_X[idx] = dna_array_dense

        t1 = time.time()
        total = t1-t0
        print('process %d peaks takes %.1f s' %(i*batch_size, total))

    f.close()

ROC_tracker = keras.metrics.AUC(curve='ROC', multi_label=True,num_thresholds=2714)
PR_metric =  keras.metrics.AUC(curve='PR', multi_label=True,num_thresholds=2714)

m_cs = tf.keras.metrics.CosineSimilarity(axis=1)

class myModell(keras.Model):
    def __init__(self,  generator1, generator2, BN):
        super(myModell, self).__init__()
        self.generator1 = generator1
        self.generator2 = generator2
        self.BN = BN

    def compile(self, g_optimizer):
        super(myModell, self).compile()
        self.g_optimizer = g_optimizer

#     def call(self, inputs, training=True):
#         output = self.generator(inputs)
#         return output

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # y_pred = self.generator(inputs = x,training = True)
            y_preprocess1 = self.generator1(inputs = x,training = True)
            y_preprocess2 = self.generator2(inputs = x,training = True)

            y_pred1 = self.BN(inputs = y_preprocess1,training = True)
            y_pred2 = self.BN(inputs = y_preprocess2,training = True)

            m_cs.update_state(y_pred1,y_pred2)
            loss1 = 1 - m_cs.result()

            y_mul = y_preprocess1*OVER1 + y_preprocess2*OVER2

            y_pred = self.BN(y_mul)

            # loss = tf.keras.losses.BinaryCrossentropy()(y, y_pred)
            loss = tf.math.reduce_mean(tf.keras.losses.BinaryCrossentropy()(y, y_pred))  + loss1
            ROC_tracker.update_state(y, y_pred)
            PR_metric.update_state(y, y_pred)
            loss_t = ROC_tracker.result()
            acc_t = PR_metric.result()

# def compute_loss(self, x, y, y_pred, sample_weight):
#     loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y))
#     loss += tf.add_n(self.losses)
#     self.loss_tracker.update_state(loss)
#     return loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        # # loss_tracker.update_state(loss)
        return {"loss": loss, "auc": loss_t,"auc_1": acc_t}

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        # return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        # y_pred = self.generator(inputs = x,training = False)
        # y_pred = self.generator(inputs = x,training = True)
        y_preprocess1 = self.generator1(inputs = x,training = True)
        y_preprocess2 = self.generator2(inputs = x,training = True)
        y_mul = y_preprocess1*OVER1 + y_preprocess2*OVER2
        y_pred = self.BN(y_mul)

        # loss = tf.keras.losses.BinaryCrossentropy()(y, y_pred)
        loss = tf.math.reduce_mean(tf.keras.losses.BinaryCrossentropy()(y, y_pred))
        ROC_tracker.update_state(y, y_pred)
        PR_metric.update_state(y, y_pred)
        loss_t = ROC_tracker.result()
        acc_t = PR_metric.result()
        ROC_tracker.reset_state()
        PR_metric.reset_state()

        # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)

#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        # # loss_tracker.update_state(loss)
        # ROC_tracker.reset_state()
        # PR_metric.reset_state()
        return {"loss": loss, "auc": loss_t,"auc_1": acc_t}

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        # return {m.name: m.result() for m in self.metrics}

    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [loss_tracker, mae_metric]

def print_memory():
    process = psutil.Process(os.getpid())
    print('cpu memory used: %.1fGB.'%(process.memory_info().rss/1e9))


# a generator to read examples from h5 file
# create a tf dataset
class generator:
    def __init__(self, file, m):
        self.file = file # h5 file for sequence
        self.m = m # csr matrix, rows as seqs, cols are cells
        self.n_cells = m.shape[1]
        self.ones = np.ones(1344)
        self.rows = np.arange(1344)

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            X = hf['X']
            for i in range(X.shape[0]):
                x = X[i]
                x_tf = sparse.coo_matrix((self.ones, (self.rows, x)),
                                               shape=(1344, 4),
                                               dtype='int8').toarray()
                y = self.m.indices[self.m.indptr[i]:self.m.indptr[i+1]]
                y_tf = np.zeros(self.n_cells, dtype='int8')
                y_tf[y] = 1
                yield x_tf, y_tf

################################
# function for post-processing #
################################
def get_cell_embedding(model, bc_model=False):
    """get cell embeddings from trained model"""
    if bc_model:
         output = model.layers[-6].get_weights()[0].transpose()
    else:
         output = model.layers[-3].get_weights()[0].transpose()
    return output

def get_intercept(model, bc_model=False):
    """get intercept from trained model"""
    if bc_model:
        output = model.layers[-6].get_weights()[1]
    else:
        output = model.layers[-3].get_weights()[1]
    return output


def imputation_Y(X, model, bc_model=False):

    if bc_model:
        new_model = tf.keras.Model(
            inputs=model.layers[0].input,
            outputs=model.layers[-5].output
        )
        Y_impute = new_model.predict(X)
    else:
        Y_impute = model.predict(X)
    return Y_impute


# perform imputation. Depth normalized.
def imputation_Y_normalize(X, model, bc_model=False, scale_method=None):
    """Perform imputation. Normalize for depth.
    Args:
        X:              feature matrix from h5.
        model:          a trained model.
    Returns:
        array:          a peak*cell imputed accessibility matrix. Sequencing depth corrected
                        for. scale_method=None, don't do any scaling of output. The raw
                        normalized output would have both positive and negative values.
                        scale_method="all_positive" scales the output by subtracting minimum value.
                        scale_method="sigmoid" scales the output by sigmoid transform.
    """
    new_model = tf.keras.Model(
        inputs=model.layers[0].input,
        outputs=model.layers[-4].output,
    )
    Y_pred = new_model.predict(X)
    w = model.layers[-3].get_weights()[0]
    accessibility_norm = np.dot(Y_pred.squeeze(), w)

    # scaling
    if scale_method == "positive":
        accessibility_norm = accessibility_norm - np.min(accessibility_norm)
    if scale_method == "sigmoid":
        accessibility_norm = np.divide(
            1,
            1 + np.exp(-accessibility_norm),
        )

    return accessibility_norm


def pred_on_fasta(fa, model, bc=False, scale_method=None):
    """Run a trained model on a fasta file.
    Args:
        fa:             fasta file to run on. Need to have a fixed size of 1344. Default
                        sequence size of trained model.
        model:          a trained model.
    Returns:
        array:          a peak*cell imputed accessibility matrix. Sequencing depth corrected for.
    """
    records = list(SeqIO.parse(fa, "fasta"))
    seqs = [str(i.seq) for i in records]
    seqs_1hot = np.array([dna_1hot(i) for i in seqs])
    pred = imputation_Y_normalize(seqs_1hot, model, bc_model=bc, scale_method=scale_method)
    return pred


def motif_score(tf, model, motif_fasta_folder, bc=False):

    fasta_motif = "%s/shuffled_peaks_motifs/%s.fasta" % (motif_fasta_folder, tf)
    fasta_bg = "%s/shuffled_peaks.fasta" % motif_fasta_folder

    pred_motif = pred_on_fasta(fasta_motif, model, bc=bc)
    pred_bg = pred_on_fasta(fasta_bg, model, bc=bc)
    tf_score = pred_motif.mean(axis=0) - pred_bg.mean(axis=0)
    tf_score = (tf_score - tf_score.mean()) / tf_score.std()
    return tf_score

# compute ism from sequence
def ism(seq_ref_1hot, model):

    new_model = tf.keras.Model(
        inputs=model.layers[0].input,
        outputs=model.layers[-4].output,
    )
    w = model.layers[-3].get_weights()[0]

    # output matrix
    m = np.zeros((model.output.shape[1], seq_ref_1hot.shape[0], seq_ref_1hot.shape[1]))

    # predication of reference seq
    latent = new_model.predict(np.array([seq_ref_1hot]))
    pred_ref = np.dot(latent.squeeze(), w)

    # compute ism
    for i in range(seq_ref_1hot.shape[0]):
        out = []
        for j in range(4):
            tmp = np.copy(seq_ref_1hot)
            tmp[i,:] = [False, False, False, False]
            tmp[i,j] = True
            out += [tmp]
        latent = new_model.predict(np.array(out))
        pred = np.dot(latent.squeeze(), w)
        m[:,i,:] = (pred - pred_ref).transpose()

    return m


class FCBlock(layers.Layer):
    def __init__(self, out_channel):
        super(FCBlock, self).__init__()
        self.Den = layers.Dense(out_channel, activation=tf.nn.gelu)
        self.Dr = layers.Dropout(0.05)

    def call(self,x):
        out = self.Den(x)
        out = self.Dr(out)

        return out

class VivitBlock(layers.Layer):
    def __init__(self, out_channel,name_1,head_nums = 8):
        super(VivitBlock, self).__init__()
        # self.c1 = layers.Conv1D(filters=out_channel, kernel_size=3, strides=2, padding="same", use_bias=False)
        # self.bn1 = layers.BatchNormalization()
        # self.r1 = GELU()
        # self.c2 = layers.Conv1D(filters=out_channel, kernel_size=3, strides=1, padding="same", use_bias=False)
        # self.bn2 = layers.BatchNormalization()
        # self.r2 = GELU()
        self.head_num = head_nums
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.MHA = MultiHeadAttention(
            head_num=head_nums,
            name=name_1,)
        self.mlp = FCBlock(out_channel)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp1 = FCBlock(out_channel)
        self.ln21 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp2 = FCBlock(out_channel)
        self.ln22 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        res = x
        x = self.ln1(x)
        if self.head_num != 0:
            x = self.MHA(x)
            res1 = x + res
            x = self.ln2(res1)
        x = self.mlp(x)
        res1 = x
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = x + res1
        return x