# all libraries / modules
import collections
import nltk
import spacy
from nltk.stem import WordNetLemmatizer 

given_text = 'Avengers: Infinity War was a 2018 American superhero film based on the Marvel Comics superhero team the Avengers. It is the 19th film in the Marvel Cinematic Universe (MCU). The running time of the movie was 149 minutes and the box office collection was around 2 billion dollars. (Source: Wikipedia)'

# tokenize given text (and make it all lower_case)
tokens = [token.lower() for token in nltk.tokenize.word_tokenize(given_text)]
print(tokens)

# get rid of non-alphabetic characters
tokens = [token for token in tokens if token.is_alpha()]
print(tokens)

# remove stop words
all_stop_words = nltk.corpus.stopwords.words('english')
tokens = [token for token in alpha_tokens if token not in all_stop_words]
print(tokens)

# counter object to see each token and its frequency
bag_of_words = collections.Counter(tokens)
print(bag_of_words)
