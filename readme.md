# Seq2seq-machine-translation

This repository provides my implementation of a Neural Machine Translation model with attention.
It closely follows the corresponding [tutorial](https://www.tensorflow.org/alpha/tutorials/sequences/nmt_with_attention) from the TensorFlow team.

## Getting Started

- File *preprocessing.py* contains functions used for preprocessing.
- File *language_index.py* contains the implementation of and index and a reverse index, starting from a given vocabulary.
- File *encoder.py* contains the encoder's implementation of the sequence-to-sequence model.
- File *decoder.py* contains the decoder's implementation of the sequence-to-sequence model.
- File *main.py* is main script, creates the dataset and trains the model.
- File *eval.py* contains functions for testing the model and plotting the attention's weights.
- File *valenti_hlt_project.pdf* contains a more in-depth project's description (in Italian).

## Installation/Dependencies

The main dependencies are:
- Python 3.5 or newer.
- TensorFlow 2.0 (highly recommended to use the GPU version).

plus a number of other python libraries. You should be able to install everything via pip (using a separate environment is highly recommended). 

## Usage

First, you need to get a dataset. The code is already tuned to used on of the [Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/), avaiable from the Tatoeba Project's page. However, you can use any dataset you want, provided it is in the right format.

Just launch the script *main.py* in the python interpreter. The script automatically trains and tests the model on the dataset.

## Getting Help

For any other additional information, you should probably check out the [original tutorial](https://www.tensorflow.org/alpha/tutorials/sequences/nmt_with_attention) from the TensorFlow team, which I closely followed for implementing this model. 

If you still have doubts, you can email me at valentiandrea@rocketmail.com.

## Contributing Guidelines  

You're free (and encouraged) to make your own contributions to this project.

## Code of Conduct

Just be nice, and respecful of each other!

## License

All source code from this project is released under the MIT license.
