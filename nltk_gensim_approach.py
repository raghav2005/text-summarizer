# all libraries / modules
import collections
import gensim
import nltk
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


given_text = 'Avengers: Infinity War was a 2018 American superhero film based on the Marvel Comics superhero team the Avengers. It is the 19th film in the Marvel Cinematic Universe (MCU). The running time of the movie was 149 minutes and the box office collection was around 2 billion dollars.'

print(alt_preprocess(given_text))
print(preprocess(given_text))


