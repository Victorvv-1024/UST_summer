import re, nltk, spacy, gensim


# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint


# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt



def clean_content(df, colname):
    """
    A function to clean the content from the dataframe
    Args:
        df (): the dataframe to be examined
        colname (string): the column from the dataframe to be cleaned

    Returns:
        list: a list of cleaned contents
    """
    contents = df[colname].values.tolist()
    
    #remove emails
    contents = [re.sub('\S*@\S*\s?', '', sent) for sent in contents]
    
    # Remove new line characters
    contents = [re.sub('\s+', ' ', sent) for sent in contents]
    
    # Remove distracting single quotes
    contents = [re.sub("\'", "", sent) for sent in contents]
    
    return contents

def tokenize(sentences):
    """
    A function to tokenize each sentence into a list of words, and removing punctuations and unnecessary characters
    Args:
        sentences (list): a list of sentences to be tokenized
    
    Returns:
        list: a list of list of tokenized sentence
    """
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    
    return list(sent_to_words(sentences))

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    A function to reduce the words to their root form
    Args:
        texts (list): list of tokenized words
        allowed_postags (list, optional):  Defaults to ['NOUN', 'ADJ', 'VERB', 'ADV'].

    Returns:
        list: list of sentence which is made of words in their root forms
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        
    return texts_out

def doc_word_matrix(content, min_df):
    """
    A function to create the document-word matrix to consider words that has occured min_df times,
    remove built-in english stopwords, covert all words to lowercase and remove any words that are below length 2
    Args:
        content (list): lemmeatized content
        min_df (int): minimum required occurences of a word 

    Returns:
        list: the doc-word matrix
    """
    
    vectorizer = CountVectorizer(analyzer='word',       
                             min_df=min_df,                    # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{2,}',  # num chars >= 2
                             # max_features=50000,             # max number of uniq words
                            )
    
    return vectorizer.fit_transform(content)

def fit_lda_model(doc_word_matrix, n_topics=10, max_iter=100, batch_size=256):
    """
    A function to fit the LDA model on the Document-Word Matrix
    Args:
        doc_word_matrix (list): the document-word matrix
        n_topics (int, optional): number of allowed topics. Defaults to 10.
        max_iter (int, optional): max number of learning iterations, epoches. Defaults to 100.
        batch_size (int, optional):  n docs in each learning iter. Defaults to 256.

    Returns:
        _type_: _description_
    """
    # build the model
    lda_model = LatentDirichletAllocation(n_topics=n_topics,     # Number of topics
                                      max_iter=max_iter,         # epochs
                                      learning_method='online',   
                                      random_state=100,          
                                      batch_size=batch_size,   
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
    # fit the model
    history = lda_model.fit_transform(doc_word_matrix)
    return history