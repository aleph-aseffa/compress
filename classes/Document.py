import numpy as np


class Document:

    def __init__(self, file_path, encoding):
        """
        Stores information about the given file.
        :param file_path: str, path to the file name to be read in.
        :param encoding: the encoding to be used for the file.
        """
        self.file_path = file_path
        self.encoding = encoding
        self.contents = None

    def store_contents(self):
        """
        Reads in the file's contents and stores it in self.contents.
        :return: None.
        """
        contents = list()
        with open(self.file_path, "r", encoding=self.encoding) as f:
            pos = 0
            for line in f:
                try:
                    sentences = line.split(".")
                    for sentence in sentences:
                        if sentence and sentence != ['\n']:
                            contents.append((sentence, pos))
                            pos += 1
                except UnicodeDecodeError:
                    pass
        self.contents = contents

    def get_embeddings(self):
        """
        Only to be called on the word embeddings file.
        Extracts the word embeddings and stores them as a dictionary in self.contents.
        :return: None.
        """
        word_embeddings = dict()

        f = open(self.file_path, encoding=self.encoding)  # GloVe word embeddings

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            word_embeddings[word] = coefs
        f.close()

        self.contents = word_embeddings
