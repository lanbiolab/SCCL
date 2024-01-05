#/bin/bash

python /tmp/SCCL/bin/preprocess.py --ad_file atac_ad.h5ad --input_fasta /tmp/hg38.fa


# python /tmp/SCCL/bin/preprocess.py --ad_file /tmp/SCCL/data/downloads/buen_ad.h5ad --input_fasta /tmp/SCCL/examples/hg19.fa
python /tmp/SCCL/bin/train.py --input_folder processed/ --epochs 1000
