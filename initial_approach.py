# all libraries / modules
import collections
import nltk
import spacy

given_text = 'Avengers: Infinity War was a 2018 American superhero film based on the Marvel Comics superhero team the Avengers. It is the 19th film in the Marvel Cinematic Universe (MCU). The running time of the movie was 149 minutes and the box office collection was around 2 billion dollars.'

# tokenize given text (and make it all lower_case)
tokens = [token.lower() for token in nltk.tokenize.word_tokenize(given_text)]

# get rid of non-alphabetic characters
tokens = [token for token in tokens if token.isalpha()]

# remove stop words
all_stop_words = nltk.corpus.stopwords.words('english')
tokens = [token for token in tokens if token not in all_stop_words]

# lemmatize all tokens
lemmatizer = nltk.stem.WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(token) for token in tokens]

# counter object to see each token and its frequency
bag_of_words = collections.Counter(tokens)

# output most common tokens
print(bag_of_words.most_common(5))

# running LDA using bag of words
# num_topics - the number of requested latent topics to be extracted from the training corpus.
# id2word - a mapping from word ids (integers) to words (strings). It is used to determine the vocabulary size, as well as for debugging and topic printing.
# workers - the number of extra processes to use for parallelization. Uses all available cores by default.
# alpha and eta - hyperparameters that affect sparsity of the document-topic (theta) and topic-word (lambda) distributions. Default value is 1/num_topics
# 	Alpha - the per document topic distribution
# 		High alpha: Every document has a mixture of all topics(documents appear similar to each other).
# 		Low alpha: Every document has a mixture of very few topics
# 	Eta - the per topic word distribution.
# 		High eta: Each topic has a mixture of most words(topics appear similar to each other).
# 		Low eta: Each topic has a mixture of few words.
# passes - the number of training passes through the corpus. For example, if the training corpus has 50,000 documents, chunksize is 10,000, passes is 2, then online training is done in 10 updates