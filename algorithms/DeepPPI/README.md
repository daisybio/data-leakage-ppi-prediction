# What is DeepPPI?

DeepPPI is a deep learning project to predict protein-protein interactions based on protein sequence only.

It is declined into PyTorch models (deprecated) and Keras models.

# How to use our Keras models?

In the 'keras' folder, run the main.py with Python. For instance, to print helps, type
`python main.py -h`

To train a model, you must give both a training set and a validation set. For instance
`python main.py -train ../data/small_1166_train.txt -val ../data/small_1166_val.txt`

In our IEEE/ACM Transactions on Computational Biology and Bioinformatics / arXiv paper, our datasets can be found in:
..* text files in `data/mirror` for our regular dataset
..* text files in `data/mirror/double` for our strict dataset

The two models from these papers can be found at `keras/models/fc2_20_2dense.py` and `keras/models/lstm32_3conv3_2dense_shared.py`. Trained weights of these models are in `keras/weights/`.

To load weights and test a model on a test set, the command is the following:
`python main.py -load <weights_file> -test <test_file>`