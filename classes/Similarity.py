import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class Similarity:

    def __init__(self, processed, embeddings):
        self.processed_text = processed.curr_text
        self.word_embeddings = embeddings.contents
        self.vectors = None
        self.matrix = None
        self.scores = None
        self.ranked_sentences = None

    def create_vectors(self):
        """
        Creates vectors for each sentence and stores them in self.vectors.
        :return: None.
        """
        # Fetch the vectors for each word in the sentence and then take the mean of those vectors.
        sentence_vectors = []
        for sentence in self.processed_text:
            if len(sentence[0]) != 0:
                v = sum([self.word_embeddings.get(w, np.zeros((100,))) for w in sentence[0].split()]) / \
                    (len(sentence[0].split()) + 0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)

            self.vectors = sentence_vectors

    def generate_similarity_matrix(self):
        """
        Creates a similarity matrix populated with the cosine similarities of the sentences in self.processed_text.
        Stored in self.matrix.
        :return: None.
        """
        similarity_matrix = np.zeros(len(self.processed_text), len(self.processed_text))
        for i in range(len(self.processed_text)):
            for j in range(len(self.processed_text)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(self.vectors[i].reshape(1, 100),
                                                                self.vectors[j].reshape(1, 100))[0, 0]
        self.matrix = similarity_matrix

    def score_similarities(self):
        """
        Generates similarity scores between the sentences using the PageRank algorithm.
        Result stored in self.scores
        :return: None.
        """
        # convert the similarity matrix into a graph.
        # the nodes are the sentences; the edges are the similarity scores between the sentences.
        nx_graph = nx.from_numpy_array(self.matrix)

        # apply PageRank algorithm to get the scores.
        scores = nx.pagerank(nx_graph)

        self.scores = scores

    def rank_sentences(self):
        """
        Sort the sentences based on their similarities.
        Result stored in self.ranked_sentences
        :return:
        """
        ranked_sentences = sorted(((self.scores[idx], sentence) for idx, sentence in enumerate(self.processed_text)),
                                  reverse=True)

        self.ranked_sentences = ranked_sentences

    def extract_summary(self, n):
        """
        Gets the top N sentences from the ranked sentences.
        :param n: int, the number of sentences to include in the summary.
        :return: str, the summary.
        """
        summary_list = [self.ranked_sentences[sentence][1] for sentence in range(n)]
        summary_list.sort(key=lambda x: x[1])

        summary = " ".join(summary_list)

        return summary


