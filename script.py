import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize

# read in the text file
text = []
with open("C:\\Users\\aleph\\Downloads\\opportunity-cost.txt", "r", encoding="utf-8") as f:
    pos = 0
    for line in f:
        try:
            sentences = line.split(".")
            for sentence in sentences:
                if sentence and sentence != ['\n']:
                    text.append((sentence, pos))
                    pos += 1
        except:
            print("skipped")
            pass

# SPLIT TEXT INTO SENTENCES

# tokenize by sentence
sentences = [(sent_tokenize(sentence[0]), sentence[1]) for sentence in text]
#sentences = [sent_tokenize(sentence[0]) for sentence in text]
sentences = [sentence for sentence in sentences if len(sentence[0]) > 0] # get rid of whitespaces
# sentences = [item for sublist in sentences for item in sublist] # flatten list

# extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8') # GloVe word embeddings
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()



# TEXT PREPROCESSING

# remove punctuations, numbers, and special characters
#clean_sentences = pd.Series(sentences).str.replace("[^a-zA-z]", " ")
clean_sentences = [(re.sub("[^a-zA-z]", " ", sentence[0][0]), sentence[1]) for sentence in sentences]

# make alphabets lowercase
#clean_sentences = [s.lower() for s in clean_sentences]
clean_sentences = [(sentence[0].lower(), sentence[1]) for sentence in clean_sentences]

# get the English language stopwords
try:
    stop_words = stopwords.words('english')
except LookupError:
    # download all the stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sent):
    new_sentence = " ".join([w for w in sent if word not in stop_words])
    return new_sentence

# remove stopwords from the sentences
clean_sentences = [(remove_stopwords(sentence[0].split()), sentence[1]) for sentence in clean_sentences]

# create vectors for our sentences.
"""
We first fetch vectors (each of size 100 elements) for the constituent words in a sentence and then take mean/average
of those vectors to arrive at a consolidated vector for the sentence.
"""
sentence_vectors = []
for sentence in clean_sentences:
    if len(sentence[0]) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in sentence[0].split()]) / (len(sentence[0].split()) + 0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)
print(sentence_vectors)
# Similarity Matrix Preparation
"""
Create an empty similarity matrix and populate it with cosine similarities of the sentences.
"""
sim_mat = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]
#exit()


# Applying PageRank Algorithm
"""
Convert the similarity matrix into a graph. The nodes of this graph will represent the sentences and the edges will
represent the similarity scores between the sentences. On this graph, we will apply the PageRank algorithm to arrive
at the sentence rankings.
"""
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

# Summary Extraction
"""
Extract the top N sentences based on their rankings for summary generation.
"""
ranked_sentences = sorted(((scores[i],s) for i, s in enumerate(sentences)), reverse=True)

# extract the top 10 sentences (this becomes the summary)
summary = [ranked_sentences[sentence][1] for sentence in range(15)]
# sort to come in order of the new article
summary.sort(key=lambda x: x[1])
for sentence in summary:
    print(sentence)















