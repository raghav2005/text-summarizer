# TOPIC-IDENTIFIER

## INFO
This is a simple tool that allows you to find the top few topics of a given paragraph or article. Currently, the only thing that happens is that the algorithm trains a model based on LDA (using gensim) to identify the 10 main topics from a dataset within the code, which is given by:
```
newsgroups_train = sklearn.datasets.fetch_20newsgroups(subset = 'train', shuffle = True)
```
This program outputs the most likely topic words in 2 ways - 1 with the relative weight of each word, and the other without the weight. TODO: I still need to make it so that the algorithm allows a user to enter their own text, and from that, the algorithm can identify and output the main topics - to do this, I'll also need to actually save the LDA model that's been trained. Currently, the algorithm takes about **1 and 1/2 minutes to run and produce an output**, so please be patient.

## REQUIREMENTS
Dependencies are in `requirements.txt` - I have used a conda environment for this project.
To install the dependencies:

- If using conda, run:
```
$ conda install --file requirements.txt
```

- If using pip, run:
```
$ pip install -r requirements.txt
```

## USAGE
The `initial_approach.py` file was just me understanding how to use the basics, so it can be ignored.
You can run the python file `nltk_gensim_approach.py`, which is essentially the main file, with:
```
python nltk_gensim_approach.py
```
