# Cracking the blackbox of deep sequence-based protein-protein interaction prediction

This repository contains all datasets and code used to show that 
sequence-based deep PPI prediction methods only achieve phenomenal 
results due to data leakage and learning from sequence similarities
and node degrees. 

We used `git-lfs` to store some of the files so make sure to install it before cloning this repository.
Most of the code can be run with our main environment ([mac](mac_env.yml), [linux](linux_env.yml)).
For PIPR, however, a custom environment ist needed ([mac](mac_env_pipr.yml), [linux](linux_env_pipr.yml)). 

![alt text](Overview%20Figure%20Results.png)

## Datasets
The original **Guo** and **Huang** datasets were obtained from `DeepFE` 
and can be found either in their [GitHub Repository](https://github.com/xal2019/DeepFE-PPI/tree/master/dataset) 
or under [`algorithms/DeepFE-PPI/dataset/11188/`](algorithms/DeepFE-PPI/dataset/11188/) (**Guo**) and [`algorithms/DeepFE-PPI/dataset/human/`](algorithms/DeepFE-PPI/dataset/human/) (**Huang**). 
The **Guo** dataset can also be found in the [PIPR respository](https://github.com/muhaochen/seq_ppi/tree/master/yeast/preprocessed) 
or under [`algorithms/seq_ppi/yeast/preprocessed/`](algorithms/seq_ppi/yeast/preprocessed/). 

The original **Du** dataset was obtained from the [original publication](https://pubs.acs.org/doi/full/10.1021/acs.jcim.7b00028) 
and can be found under [`Datasets_PPIs/Du_yeast_DIP/`](Datasets_PPIs/Du_yeast_DIP/).

The **Pan** dataset can be obtained from the [original publication](http://www.csbio.sjtu.edu.cn/bioinf/LR_PPI/Data.htm) 
and from the [PIPR Repository](https://github.com/muhaochen/seq_ppi/tree/master/sun/preprocessed). 
It is in [`algorithms/seq_ppi/sun/preprocessed/`](algorithms/seq_ppi/sun/preprocessed/). 

The **Richoux** datasets were obtained from their [Gitlab](https://gitlab.univ-nantes.fr/richoux-f/DeepPPI/-/tree/master/data).
The **regular** dataset is in [`algorithms/DeepPPI/data/mirror/`](algorithms/DeepPPI/data/mirror/), the **strict** one in
[`algorithms/DeepPPI/data/mirror/double/`](algorithms/DeepPPI/data/mirror/double/). 

All original datasets were rewritten into the format used by SPRINT and split 
into train and test with [`algorithms/SPRINT/create_SPRINT_datasets.py`](algorithms/SPRINT/create_SPRINT_datasets.py).
They are in in [`algorithms/SPRINT/data/original`](algorithms/SPRINT/data/original).
This script was also used to **rewire** and split the datasets (`generate_RDPN`) (-> [`algorithms/SPRINT/data/rewired`](algorithms/SPRINT/data/rewired)).
Before you run this script, you have to run [`compute_sim_matrix.py`](algorithms/Custom/compute_sim_matrix.py). 

### Partitions

The [human](Datasets_PPIs/SwissProt/human_swissprot.fasta) and [yeast](Datasets_PPIs/SwissProt/yeast_swissprot.fasta) proteomes were downloaded from Uniprot and sent to the 
team of SIMAP2. They sent back the similarity data which we make available under
[https://doi.org/10.6084/m9.figshare.21510939](https://doi.org/10.6084/m9.figshare.21510939) (`submatrix.tsv.gz`). 
Download this and unzip it in `network_data/SIMAP2`.

We preprocessed this data in order to give it to the KaHIP kaffpa algorithm with [simap_preprocessing.py](simap_preprocessing.py):

1. We separated the file to obtain only human-human and yeast-yeast protein similarities
2. We converted the edge lists to networks and converted the Uniprot node labels to integer labels because KaHIP needs `METIS` files as input. These files can only handle integer node labels
3. We exported the networks as `METIS` files with bitscores as edge weights: [human](network_data/SIMAP2/human_networks/only_human_bitscore.graph), [yeast](network_data/SIMAP2/yeast_networks/only_yeast_bitscore.graph)

If you're using a **Mac**, you can use our compiled KaHIP version. On **Linux**, make sure you have OpenMPI installed and run the following commands: 
```
rm -r KaHIP
git clone https://github.com/KaHIP/KaHIP
cd KaHIP/
./compile_withcmake.sh
cd ..
```
Then, feed the METIS files to the KaHIP kaffpa algorithm with the following commands: 
```
./KaHIP/deploy/kaffpa ./network_data/SIMAP2/human_networks/only_human_bitscore.graph --seed=1234 --output_filename="./network_data/SIMAP2/human_networks/only_human_partition_bitscore.txt" --k=2 --preconfiguration=strong
./KaHIP/deploy/kaffpa ./network_data/SIMAP2/yeast_networks/only_yeast_bitscore.graph --seed=1234 --output_filename="./network_data/SIMAP2/yeast_networks/only_yeast_partition_bitscore.txt" --k=2 --preconfiguration=strong
```

The output files containing the partitioning was mapped back to the original UniProt IDs in [kahip.py](kahip.py). Nodelists: [human](network_data/SIMAP2/human_networks/only_human_partition_nodelist.txt), [yeast](network_data/SIMAP2/yeast_networks/only_yeast_partition_nodelist.txt).

The PPIs from the 6 original datasets were then split according to the KaHIP partitions into blocks
Inter, Intra-0, and Intra-1 with [rewrite_datasets.py](rewrite_datasets.py) and are in [`algorithms/SPRINT/data/partitions`](algorithms/SPRINT/data/partitions).

### Gold Standard Dataset
We wanted our gold standard dataset to be split into training, validation, and testing. 
There should be no overlaps between the three datasets and a minimum amount of sequence similarity 
so that the methods can learn more complex features. 
Hence, we partitioned the human proteome into three parts by running: 
```
./KaHIP/deploy/kaffpa ./network_data/SIMAP2/human_networks/only_human_bitscore.graph --seed=1234 --output_filename="./network_data/SIMAP2/human_networks/only_human_partition_3_bitscore.txt" --k=3 --preconfiguration=strong
```
Then, the Hippie v2.3 database was downloaded from their [website](http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/).
The dataset was split intro training, validation, and testing using the partition. Negative PPIs were 
sampled randomly, node degrees of proteins from the positive dataset were preserved in expectation in the 
negative dataset. The resulting blocks Intra-0, Intra-1, and Intra-2 were redundancy-reduced using CD-HIT. 
CD-HIT was cloned from their [GitHub](https://github.com/weizhongli/cdhit.git) and built following the instructions given there.
The datasets were redundancy reduced at 40% pairwise sequence similarity by first exporting their fasta sequences and then running:
```
./cdhit/cd-hit -i Datasets_PPIs/Hippiev2.3/Intra_0.fasta -o sim_intra0.out -c 0.4 -n 2
./cdhit/cd-hit -i Datasets_PPIs/Hippiev2.3/Intra_1.fasta -o sim_intra1.out -c 0.4 -n 2
./cdhit/cd-hit -i Datasets_PPIs/Hippiev2.3/Intra_2.fasta -o sim_intra2.out -c 0.4 -n 2
```
Redundancy was also reduced between the datasets: 
```
./cdhit/cd-hit-2d -i Datasets_PPIs/Hippiev2.3/Intra_0.fasta -i2 Datasets_PPIs/Hippiev2.3/Intra_1.fasta -o Datasets_PPIs/Hippiev2.3/sim_intra0_intra_1.out -c 0.4 -n 2
./cdhit/cd-hit-2d -i Datasets_PPIs/Hippiev2.3/Intra_0.fasta -i2 Datasets_PPIs/Hippiev2.3/Intra_2.fasta -o Datasets_PPIs/Hippiev2.3/sim_intra0_intra_2.out -c 0.4 -n 2
./cdhit/cd-hit-2d -i Datasets_PPIs/Hippiev2.3/Intra_1.fasta -i2 Datasets_PPIs/Hippiev2.3/Intra_2.fasta -o Datasets_PPIs/Hippiev2.3/sim_intra1_intra_2.out -c 0.4 -n 2
```
Then, the redundant sequences were extracted from the output files
```
less Datasets_PPIs/Hippiev2.3/sim_intra0.out.clstr| grep -E '([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}).*%$'|cut -d'>' -f2|cut -d'.' -f1 > Datasets_PPIs/Hippiev2.3/redundant_intra0.txt
less Datasets_PPIs/Hippiev2.3/sim_intra1.out.clstr| grep -E '([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}).*%$'|cut -d'>' -f2|cut -d'.' -f1 > Datasets_PPIs/Hippiev2.3/redundant_intra1.txt
less Datasets_PPIs/Hippiev2.3/sim_intra2.out.clstr| grep -E '([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}).*%$'|cut -d'>' -f2|cut -d'.' -f1 > Datasets_PPIs/Hippiev2.3/redundant_intra2.txt

less Datasets_PPIs/Hippiev2.3/sim_intra0_intra_1.out.clstr| grep -E '([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}).*%$'|cut -d'>' -f2|cut -d'.' -f1 > Datasets_PPIs/Hippiev2.3/redundant_intra01.txt
less Datasets_PPIs/Hippiev2.3/sim_intra0_intra_2.out.clstr| grep -E '([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}).*%$'|cut -d'>' -f2|cut -d'.' -f1 > Datasets_PPIs/Hippiev2.3/redundant_intra02.txt
less Datasets_PPIs/Hippiev2.3/sim_intra1_intra_2.out.clstr| grep -E '([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}).*%$'|cut -d'>' -f2|cut -d'.' -f1 > Datasets_PPIs/Hippiev2.3/redundant_intra12.txt
```
and filtered out of training, validation, and testing. 
All Python code for this task can be found in [create_gold_standard.py](create_gold_standard.py). 

## Methods

### Custom baseline ML methods
Our 6 implemented baseline ML methods are implemented in [`algorithms/Custom/`](algorithms/Custom). 

1. We converted the SIMAP2 similarity table into an all-against-all similarity matrix in [`algorithms/Custom/compute_sim_matrix.py`](algorithms/Custom/compute_sim_matrix.py).
2. We reduced the dimensionality of this matrix via PCA, MDS, and node2vec in [`algorithms/Custom/compute_dim_red.py`](algorithms/Custom/compute_dim_red.py):
   1. PCA [human](algorithms/Custom/data/human_pca.npy), [yeast](algorithms/Custom/data/yeast_pca.npy)
   2. MDS [human](algorithms/Custom/data/human_mds.npy), [yeast](algorithms/Custom/data/yeast_mds.npy)
   3. For node2vec, we first converted the similarity matrix into a network and exported its edgelist ([human](algorithms/Custom/data/human.edgelist), [yeast](algorithms/Custom/data/yeast.edgelist)) and nodelist ([human](algorithms/Custom/data/human.nodelist), [yeast](algorithms/Custom/data/yeast.nodelist)). Then, we called node2vec

If you have a **Mac**, you can use the precompiled node2vec binaries. If you have a **Linux**, follow the following steps: 
```
rm -r snap
git clone https://github.com/snap-stanford/snap.git
cd snap
make all
cd ..
```
Then, call node2vec with
```
cd snap/examples/node2vec
./node2vec -i:../../../algorithms/Custom/data/yeast.edgelist -o:../../../algorithms/Custom/data/yeast.emb
./node2vec -i:../../../algorithms/Custom/data/human.edgelist -o:../../../algorithms/Custom/data/human.emb
```
The RF and SVM are implemented in [algorithms/Custom/learn_models.py](algorithms/Custom/learn_models.py). 
All tests are executed in [algorithms/Custom/run.py](algorithms/Custom/run.py).
Results were saved to the [results folder](algorithms/Custom/results).
### DeepFE
The code was pulled from their [GitHub Repository](https://github.com/xal2019/DeepFE-PPI/) and updated to the current tensorflow version.
All tests are run via [the shell slurm script](algorithms/DeepFE-PPI/run_DeepFE.sh) or [algorithms/DeepFE-PPI/train_all_datasets.py](algorithms/DeepFE-PPI/train_all_datasets.py).
Results were saved to the [results folder](algorithms/DeepFE-PPI/result).

### Richoux-FC and Richoux-LSTM
The code was pulled from their [Gitlab Repository](https://gitlab.univ-nantes.fr/richoux-f/DeepPPI). 
All tests can be run via [the shell slurm script](algorithms/DeepPPI/keras/run_DeepPPI.sh) or [algorithms/DeepPPI/keras/train_all_datasets.py](algorithms/DeepPPI/keras/train_all_datasets.py).
Results were saved to the [results folder](algorithms/DeepPPI/keras/results_custom).

### PIPR
The code was pulled from their [GitHub Repository](https://github.com/muhaochen/seq_ppi) and updated to the current tensorflow version.
Activate the **PIPR environment** for running all PIPR code! 
All tests are run via [the shell slurm script](algorithms/seq_ppi/binary/model/lasagna/run_PIPR.sh) or [algorithms/seq_ppi/binary/model/lasagna/train_all_datasets.py](algorithms/seq_ppi/binary/model/lasagna/train_all_datasets.py).
Results were saved to the [results folder](algorithms/seq_ppi/binary/model/lasagna/results).

### SPRINT
The code was pulled from their [GitHub Repository](https://github.com/lucian-ilie/SPRINT). 
You need a g++ compiler and the boost library ([http://www.boost.org/](http://www.boost.org/)) to compile the source code. 

After downloading boost, move it to a fitting directory like `/usr/local/`.
Edit the [makefile](algorithms/SPRINT/makefile) and adapt the path to boost (`-I /usr/local/boost_1_80_0`).
Then run 
```
cd algorithms/SPRINT
mkdir bin
make predict_interactions_serial
make compute_HSPs_serial 
```
The yeast proteome fasta file was first transformed such that each sequence only occupies one line [rewrite_yeast_fasta.py](algorithms/SPRINT/rewrite_yeast_fasta.py).

Then, the proteome was preprocessed with [compute_yeast_HSPs.sh](algorithms/SPRINT/compute_yeast_HSPs.sh).

The preprocessed human proteome was downloaded from the [SPRINT website](https://www.csd.uwo.ca/~ilie/SPRINT/) (precomputed similarities). 
After downloading the data, move it to the `HSP` folder in `algorithms/SPRINT`. 

Then tests are run via shell slurm scripts: [original](algorithms/SPRINT/run_SPRINT_original.sh), [rewired](algorithms/SPRINT/run_SPRINT_rewired.sh), [partitions](algorithms/SPRINT/run_SPRINT_custom.sh).
Results were saved to the [results folder](algorithms/SPRINT/results).
AUCs and AUPRs were calculated with [algorithms/SPRINT/results/calculate_scores.py](algorithms/SPRINT/results/calculate_scores.py).

## Visualizations
All visualizations were made with the R scripts in [visualizations](visualizations).
Plots were saved to [visualizations/plots](visualizations/plots).





