# Cracking the blackbox of deep sequence-based protein-protein interaction prediction

This repository contains all datasets and code used to show that 
sequence-based deep PPI prediction methods only achieve phenomenal 
results due to data leakage and learning from sequence similarities
and node degrees. 

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

All original datasets were rewritten into the format used by SPRINT with
[`algorithms/SPRINT/create_SPRINT_datasets.py`](algorithms/SPRINT/create_SPRINT_datasets.py)
and can be found in [`algorithms/SPRINT/data/original`](algorithms/SPRINT/data/original).
This script was also used to **rewire** the datasets (`generate_RDPN`) (-> [`algorithms/SPRINT/data/rewired`](algorithms/SPRINT/data/rewired)).

### Partitions

The human and yeast proteomes were downloaded from Uniprot and sent to the 
team of SIMAP2. They sent back the similarity data which we make available under
[https://syncandshare.lrz.de/getlink/fi5AJEoSLB1DrXjxAzBne7/](https://syncandshare.lrz.de/getlink/fi5AJEoSLB1DrXjxAzBne7/). 

From that, we computed METIS files [simap_preprocessing.py](simap_preprocessing.py), 
which we fed to the KaHIP kaffpa algorithm with the following command: 
```
./KaHIP/deploy/kaffpa ./network_data/SIMAP2/only_human_bitscore.graph --seed=1234 --output_filename=--output_filename="./network_data/SIMAP2/only_human_partition_bitscore.txt" --k=2 --preconfiguration=strong
```
(analogous for yeast). 

The output file containing the partitioning was mapped back to the original UniProt IDs in [kahip.py](kahip.py).

The PPIs from the 6 original datasets were then split according to the KaHIP partitions into blocks
Inter, Intra-0, and Intra-1 with [rewrite_datasets.py](rewrite_datasets.py).

## Methods





