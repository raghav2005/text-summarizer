# all libraries / modules
import collections
import gensim
import nltk
import sklearn
import spacy

# lemmatize and stemming
def lemmatize_stemming(text):
	stemmer = nltk.stem.SnowballStemmer('english')
	return stemmer.stem(nltk.stem.WordNetLemmatizer().lemmatize(text, pos='v'))

# alternative lemmatize and stemming
def alt_lemmatize_stemming(given_text):
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

	# convert tokens list to string
	tokens_str = ''
	for i in range(len(tokens)):
		if i != len(tokens) - 1:
			tokens_str += tokens[i] + ' '
		else:
			tokens_str += tokens[i]

	return lemmatize_stemming(tokens_str)

# tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

# alternative tokenize and lemmatize
def alt_preprocess(given_text):
    result = []
    for token in gensim.utils.simple_preprocess(given_text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(alt_lemmatize_stemming(token))
            
    return result


# document text
given_text = 'This disk has failed many times. I would like to get it replaced.'

# we will be using the actual approach - not the alt
words = []
for word in given_text.split(' '):
    words.append(word)

# preprocessing
print('Original document:', words)
print('\n\nTokenized and lemmatized document:', preprocess(given_text), '\n\n')

# training
newsgroups_train = sklearn.datasets.fetch_20newsgroups(subset = 'train', shuffle = True)
newsgroups_test = sklearn.datasets.fetch_20newsgroups(subset = 'test', shuffle = True)

processed_docs = []

for doc in newsgroups_train.data:
    processed_docs.append(preprocess(doc))

print(processed_docs[ : 2], '\n\n')

# bag of words
dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

# remove very rare and very common
dictionary.filter_extremes(no_below = 15, no_above = 0.1, keep_n = 100000)
# create bag of words model for each doc
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# preview bag of words for sample preprocessed document
document_num = 20
bow_doc_x = bow_corpus[document_num]

for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], dictionary[bow_doc_x[i][0]], bow_doc_x[i][1]))


