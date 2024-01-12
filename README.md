# SCCL
With the advance of single-cell assay for transposase-accessible chromatin sequencing technologies (scATAC-seq), it is able to assess the accessibility of single-cell chromatin and gain insights into the process of gene regulation. However, the scATAC data contains distinct characteristics such as sparsity and high dimensionality, which often pose challenges in the downstream analysis. In this paper, we introduce a contrastive learning method (SCCL) for modeling scATAC data. The SCCL designs two distinct encoders to extract local and global features from the original data, respectively. In addition, an improved contrastive learning method is utilized to reduce the redundancy of the feature. Further, the local and global fea-tures are fused to obtain reliable features. Finally, the decode is used to generate binary accessibility. We conduct the ex-periment on various real datasets, and the results demonstrate its superiority over other state-of-the-art methods in cell clus-ter, transcription factor (TF) activity inference and batch correction.

Author: Wei Lan, Weihao Zhou, Qingfeng Chen, Ruiqing Zheng, Min Li, Yi Pan, Yi-Ping Phoebe Chen

# Environment Requirement
+ anndata==0.8.0
+ Bio==1.5.8
+ ConfigArgParse==1.5.3
+ h5py==3.6.0
+ jupyterlab==3.3.3
+ leidenalg==0.8.9
+ matplotlib==3.5.3
+ numba==0.55.1
+ numpy==1.21.5
+ pandas==1.3.5
+ psutil==5.9.0
+ pysam==0.19.0
+ scanpy==1.9.3
+ scipy==1.7.3
+ seaborn==0.11.2
+ setuptools==58.0.4
+ tensorflow==2.8.0

# Dataset
We use two datasets including Buenrostro2018 and Pe-ripheral Blood Mononuclear Cells (PBMC) to evaluate the performance of model. The Buenrostro2018 contains 2034 cells, 10 types of cells, 1884 genes and 6442 high-quality fragments per cell. The PBMC dataset is obtained from Chen et al. It contains 2714 cells, 1791 genes and 14479 high-quality frag-ments per cell.
+ Buenrostro2018 https://www.ncbi.nlm.nih.gov/gds?linkname=pubmed_gds&from_uid=29706549
+ PBMC           https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_web_summary.html

# Running
The way to run the following commands:
```
./train.sh
```