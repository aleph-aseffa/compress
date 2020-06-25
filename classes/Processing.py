from nltk.tokenize import sent_tokenize
import re
import nltk
from nltk.corpus import stopwords


class Processing:

    def __init__(self, doc):
        self.curr_text = doc.contents
        self.stop_words = None
        self.get_stop_words()

    def get_stop_words(self):
        """
        Downloads the English stopwords from the NLTK library and stores it in self.stop_words.
        :return: None.
        """
        try:
            stop_words = stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
            stop_words = stopwords.words('english')

        self.stop_words = stop_words

    def tokenize_by_sentence(self):
        """
        Tokenizes each sentence in self.curr_text.
        :return: None
        """
        # tokenize by sentence
        sentences = [(sent_tokenize(sentence[0]), sentence[1]) for sentence in self.curr_text]
        # get rid of whitespaces
        sentences = [sentence for sentence in sentences if len(sentence[0]) > 0]
        self.curr_text = sentences

    def remove_formatting(self):
        """
        Removes formatting from self.curr_text.
        :return: None.
        """
        # remove punctuations, numbers, and special characters
        clean_sentences = [(re.sub("[^a-zA-z]", " ", sentence[0][0]), sentence[1]) for sentence in self.curr_text]

        # make alphabets lowercase
        clean_sentences = [(sentence[0].lower(), sentence[1]) for sentence in clean_sentences]

        self.curr_text = clean_sentences

    def remove_stopwords(self):
        """
        Removes stopwords from self.curr_text.
        :return: None.
        """
        def join(s):
            new_s = " ".join([word for word in s if s not in self.stop_words])
            return new_s
        without_stopwords = [(join(sentence[0].split()), sentence[1]) for sentence in self.curr_text]

        self.curr_text = without_stopwords

    def get_processed_text(self):
        """
        Returns the processed text.
        :return: str: the processed text.
        """
        return self.curr_text
