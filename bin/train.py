#!/usr/bin/env python
import anndata
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil
import math
import pickle
import seaborn as sns
import scipy
import configargparse
import sys
import gc
from datetime import datetime
from SCCL.utils import *
from SCCL.basenji_utils import *
from tensorflow.keras import layers
from tensorflow import keras

from keras_multi_head import MultiHeadAttention


def make_parser():
    parser = configargparse.ArgParser(
        description="train on scATAC data")
    parser.add_argument('--input_folder', type=str,
                        help='folder contains preprocess files. The folder should contain: train_seqs.h5, test_seqs.h5, val_seqs.h5, splits.h5, ad.h5ad')
    parser.add_argument('--bottleneck', type=int, default=32,
                        help='size of bottleneck layer. Default to 32')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size. Default to 128')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate. Default to 0.01')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train. Default to 1000.')
    parser.add_argument('--out_path', type=str, default='output',
                        help='Output path. Default to ./output/')
    parser.add_argument('--print_mem', type=bool, default=True,
                        help='whether to output cpu memory usage.')

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    preprocess_folder = args.input_folder
    bottleneck_size = args.bottleneck
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    out_dir = args.out_path
    print_mem = args.print_mem

    train_data = '%s/train_seqs.h5' % preprocess_folder
    val_data = '%s/val_seqs.h5' % preprocess_folder
    split_file = '%s/splits.h5' % preprocess_folder
    ad = anndata.read_h5ad('%s/ad.h5ad' % preprocess_folder)
    n_cells = ad.shape[0]

    # convert to csr matrix
    with h5py.File(split_file, 'r') as hf:
        train_ids = hf['train_ids'][:]
        val_ids = hf['val_ids'][:]

    m = ad.X.tocoo().transpose().tocsr()
    if print_mem:
        print_memory()  # memory usage

    del ad
    gc.collect()

    m_train = m[train_ids, :]
    m_val = m[val_ids, :]
    del m
    gc.collect()

    # generate tf.datasets
    train_ds = tf.data.Dataset.from_generator(
        generator(train_data, m_train),
        output_signature=(
            tf.TensorSpec(shape=(1344, 4), dtype=tf.int8),
            tf.TensorSpec(shape=(n_cells), dtype=tf.int8),
        )
    ).shuffle(2000, reshuffle_each_iteration=True).batch(128).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        generator(val_data, m_val),
        output_signature=(
            tf.TensorSpec(shape=(1344, 4), dtype=tf.int8),
            tf.TensorSpec(shape=(n_cells), dtype=tf.int8),
        )
    ).batch(128).prefetch(tf.data.AUTOTUNE)

    Generator1 = keras.Sequential(
        [
            keras.Input(shape=(1344, 4), ),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            # layers.Dense(7 * 7 * 128),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Reshape((7, 7, 128)),
            # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            StochasticReverseComplement(),
            StochasticShift(3),
            GELU(),
            layers.Conv1D(
                filters=288,
                kernel_size=17,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones", ),
            layers.MaxPool1D(pool_size=3, padding="same"),
            GELU(),
            FCBlock(288),
            # layers.Conv1D(
            #     filters=288,
            #     kernel_size=5,
            #     strides=1,
            #     padding="same",
            #     use_bias=False,
            #     dilation_rate=1,
            #     kernel_initializer="he_normal",
            #     kernel_regularizer=tf.keras.regularizers.l2(0),
            # ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Conv1D(
                filters=323,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Conv1D(
                filters=362,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Conv1D(
                filters=406,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Conv1D(
                filters=456,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            # layers.Conv1D(
            #     filters=512,
            #     kernel_size=5,
            #     strides=1,
            #     padding="same",
            #     use_bias=False,
            #     dilation_rate=1,
            #     kernel_initializer="he_normal",
            #     kernel_regularizer=tf.keras.regularizers.l2(0),
            # ),
            FCBlock(512),
            # MultiHeadAttention(
            #     head_num=8,
            #     name="name_1",),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
        ],
        name="Generator1", )

    Generator2 = keras.Sequential(
        [
            keras.Input(shape=(1344, 4), ),
            # We want to generate 128 coefficients to reshape into a 7x7x128 map
            # layers.Dense(7 * 7 * 128),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Reshape((7, 7, 128)),
            # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            StochasticReverseComplement(),
            StochasticShift(3),
            GELU(),
            layers.Conv1D(
                filters=288,
                kernel_size=17,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones", ),
            layers.MaxPool1D(pool_size=3, padding="same"),
            GELU(),
            FCBlock(288),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Conv1D(
                filters=323,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Conv1D(
                filters=362,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Conv1D(
                filters=406,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Conv1D(
                filters=456,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            FCBlock(512),
            MultiHeadAttention(
                head_num=8,
                name="name_1", ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.MaxPool1D(pool_size=2, padding="same"),
            MultiHeadAttention(
                head_num=8,
                name="MHA1", ),
        ],
        name="Generator2", )

    BottleNeck = keras.Sequential(
        [

            GELU(),
            layers.Conv1D(
                filters=256,
                kernel_size=5,
                strides=1,
                padding="same",
                use_bias=False,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            # layers.MaxPool1D(pool_size=2, padding="same"),
            GELU(),
            layers.Reshape((1, 1792,)),
            layers.Dense(
                units=32,
                use_bias=(not True),
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l1_l2(0, 0),
            ),
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones"),
            layers.Dropout(rate=0.2),
            GELU(),
            layers.Dense(
                units=n_cells,
                use_bias=True,
                activation="sigmoid",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l1_l2(0, 0),
            ),
            layers.Flatten()
        ],
        name="BottleNeck", )

    tf.keras.backend.set_learning_phase(True)
    model = myModell(generator1=Generator1, generator2=Generator2, BN=BottleNeck)
    filepath = '%s/best_model.h5' % out_dir

    callbacks = [
        tf.keras.callbacks.TensorBoard(out_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True,
                                           save_weights_only=True, monitor='auc', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='auc', min_delta=1e-6,
                                         mode='max', patience=50, verbose=1),
    ]

    model.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.95, beta_2=0.9995),
        # metrics=[tf.keras.metrics.AUC(curve='ROC', multi_label=True),
        #         tf.keras.metrics.AUC(curve='PR', multi_label=True)]
    )

    # tensorboard
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds
    )
    pickle.dump(history.history, open('%s/history.pickle' % out_dir, 'wb'))


if __name__ == "__main__":
    main()
