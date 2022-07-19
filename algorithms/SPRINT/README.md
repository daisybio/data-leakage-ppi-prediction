## SPRINT (Scoring PRotein INTeractions)

### Citation 
Y. Li, L. Ilie, SPRINT: Ultrafast protein-protein interaction prediction of the entire human interactome, BMC Bioinformatics 18 (2017) 485.

Y. Li, L. Ilie, Predicting Protein–Protein Interactions Using SPRINT, In Protein-Protein Interaction Networks (pp. 1-11). Humana, New York, NY.
### Author: 

Yiwei Li, Lucian Ilie

Contact:

Yiwei Li (yli922@uwo.ca), Lucian Ilie (ilie@uwo.ca)

Department of Computer Science

The University of Western Ontario, London, Ontario, Canada

### Description

SPRINT consists of two components: compute_HSPs and predict_interactions. 

`Compute_HSPs` computes the similar sub-sequences among all input proteins. 

`Predict_interactions` predicts interactions based on the computed similarities and input training data.

Since pre-computing human HSPs takes about two days on 12 cores, we provide the computed HSPs of the entire human proteome. The human protein sequences are from uniprot (http://www.uniprot.org/). The number of human proteins that we used to compute theses HSPs is 20160.

### Installation

You need g++ compiler, OpenMP(http://openmp.org/wp/) library(if you want to run the program in parallel), and boost(http://www.boost.org/) library(only for compute_HSPs) 

1. Enter the directory of SPRINT

2. 
 * to compile the serial version of compute_HSPs type "make compute_HSPs_serial" (require g++, boost library)
 * to compile the parallel version of compute_HSPs type "make compute_HSPs_parallel" (require g++, boost library, and OpenMP)
 * to compile the serial version of predict_interactions type "make predict_interactions_serial" (only require g++ under any Unix-like environment)
 * to compile the parallel version of predict_interactions type "make predict_interactions_parallel" (require g++ and OpenMP)

### Toy Examples

Please go the the directory SPRINT/toy_example to see the inputs format of SPRINT. 

### Input files

The directory SPRINT/toy_example contains input files as a toy dataset for SPRINT.

They are:

- A protein sequence file: protein_sequences.seq
- A positive training file: train_positive.txt (optional)
- Two testing files: test_positive.txt and test_negative.txt


### Input files format

1. protein sequence file: the file that contains the entire proteome sequences for an organism
 For each protein, there are two lines.

 The first line contain a ">" followed by the protein name.

 The second line is the protein sequence. Note here each sequence has to be in the same line. SPRINT could handle the 23 amino acids in the PAM120 matrix(http://blast.advbiocomp.com/blast-1.3/blast/PAM120) plus 'U' and 'O'. If the input sequences contain other than those, the program will exit and notify the user.

2. Positive training file: the file that contains known interactions

 Each known interaction forms one line.

 The format is: < Protein1_name > < Protein2_name >

3. Testing files (optional)	: the files that contains testing interactions	

 Each known interaction forms one line.

 The format is: < Protein1_name > < Protein2_name >

### How to run SPRINT

SPRINT consists of two parts: compute_HSPs and predict_interactions.

1. compute_HSPs computes the HSPs in a certain organism.

 In order to run compute_HSPs type "bin/compute_HSPs" followed by options:

 Options:

 -p < protein_file > (required)

 -h < hsp_output_file_name > (required)

 -Thit < an integer, the threshold Thit > (optional, default: 15) 

 -Tsim < an integer, the threshold Tsim > (optional, default: 35) 

 -M < an integer, Scoring matrix. 1: PAM120, 2: BLOSUM80, 3: BLOSUM62> (optional, default: PAM120)
 
 -add <new_protein_file previous_HSP_file_name> (optional, if new protein sequences are added and only HSPs in those sequences will be computed. New HSPs will be appended to previous_HSP_file. hsp_output_file_name will be ignored)

2. predict_interactions calculates the scores for given pairs or perform the entire interactome prediction.
 In order to run predict_interactions type "bin/predict_interactions" followed by options:
 
 -p < protein_file > (required)

 -h < hsp_file > (required)

 -Thc < an integer, the threshold to be considered a high count > (optional, default 40)

 -tr < training_file > (required)

 -pos < positive_testing_file > (optional)

 -neg < negative_testing_file > (optional) 

 -o < output_file > (required)

 -e (if you need to perform the entire proteome prediction) (optional)

### Example

The commands for running SPRINT on the toy dataset in SPRINT/toy_example are given below.

1. compute_HSPs

 ```
 bin/compute_HSPs -p toy_example/protein_sequences.seq -h HSP/hsps_toy_example 
 ```
 The above command creates a file hsps_toy_example in the directory SPRINT/HSP

2. predict_interactions (use either 2.1 or 2.2)

 2.1 calculate the scores for the pairs in toy_example/test_positive.txt and toy_example/test_negative.txt.
 ```
 bin/predict_interactions -p toy_example/protein_sequences.seq -h HSP/hsps_toy_example -tr toy_example/train_positive.txt -pos toy_example/test_positive.txt -neg toy_example/test_negative.txt -o toy_example/result_test.txt
 ```
 2.2 perform the entire interactome prediction.
```
bin/predict_interactions -p toy_example/protein_sequences.seq -h HSP/hsps_toy_example -tr toy_example/train_positive.txt -e -o toy_example/result_interactome.txt
```


