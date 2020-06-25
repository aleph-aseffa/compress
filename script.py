from classes import Document as doc
from classes import Processing as proc
from classes import Similarity as sim


def main():
    """
    Used to run the entire program.
    :return: None.
    """
    # document to be summarized
    article = doc.Document("C:\\Users\\aleph\\Downloads\\opportunity-cost.txt", "utf-8")
    article.store_contents()

    # word embeddings
    embeddings = doc.Document("glove.6B.100d.txt", "utf-8")
    embeddings.get_embeddings()

    # process the document
    article_processor = proc.Processing(article)
    article_processor.tokenize_by_sentence()
    article_processor.remove_formatting()
    article_processor.get_stop_words()
    article_processor.remove_stopwords()

    # analyze the document
    analyzer = sim.Similarity(article, embeddings)
    analyzer.create_vectors()
    analyzer.generate_similarity_matrix()
    analyzer.score_similarities()
    analyzer.rank_sentences()

    # extract and display summary
    summary = analyzer.extract_summary(15)
    print(summary)











