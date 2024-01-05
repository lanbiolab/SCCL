#!/usr/bin/env python
import os
import h5py
import anndata
import configargparse
import numpy as np
import pandas as pd
import scanpy as sc
from SCCL.utils import *


def make_parser():
    parser = configargparse.ArgParser(
        description="Preprocess anndata to generate inputs.")
    parser.add_argument('--ad_file', type=str,
                        help='Input scATAC anndata. .var must have chr, start, end columns. anndata.X must be in csr format.')
    parser.add_argument('--input_fasta', type=str,
                        help='Genome fasta file.')
    parser.add_argument('--out_path', type=str, default='./processed',
                        help='Output path. Default to ./processed/')

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    input_ad = args.ad_file
    input_fasta = args.input_fasta
    output_path = args.out_path

    ad = anndata.read_h5ad(input_ad)

    # sample cells
    data_path = '.'
    os.makedirs(output_path, exist_ok=True)
    seq_len = 1344

    ad.write('%s/ad.h5ad' % output_path)
    ad.var.loc[:, ['chr', 'start', 'end']].to_csv('%s/peaks.bed' % output_path, sep='\t', header=False, index=False)

    train_ids, test_ids, val_ids = split_train_test_val(np.arange(ad.shape[1]))
    f = h5py.File('%s/splits.h5' % output_path, "w")
    f.create_dataset("train_ids", data=train_ids)
    f.create_dataset("test_ids", data=test_ids)
    f.create_dataset("val_ids", data=val_ids)
    f.close()

    ad_train = ad[:, train_ids]
    ad_test = ad[:, test_ids]
    ad_val = ad[:, val_ids]

    make_h5_sparse(ad, '%s/all_seqs.h5' % output_path, input_fasta)
    make_h5_sparse(ad_train, '%s/train_seqs.h5' % output_path, input_fasta)
    make_h5_sparse(ad_test, '%s/test_seqs.h5' % output_path, input_fasta)
    make_h5_sparse(ad_val, '%s/val_seqs.h5' % output_path, input_fasta)


if __name__ == "__main__":
    main()
