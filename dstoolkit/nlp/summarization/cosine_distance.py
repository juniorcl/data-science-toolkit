import nltk
import spacy
import heapq

import numpy    as np
import networkx as nx

from collections import Counter


class CosineDistanceSummarizer():

    def __init__(self, n_sentences, lemma=True, spacy_model="pt_core_news_sm"):
        
        self.n_sentences = n_sentences

        self._nlp = spacy.load(spacy_model)

        self._lemma = lemma

    def _tokenize(self, text):

        return nltk.sent_tokenize(text)

    def _preprocess_sentences(self, sentences):
            
        return [
            " ".join(
                token.lemma_ if self._lemma else token.text_
                for token in self._nlp(sentence.lower())
                if not token.is_stop and not token.is_punct
            )
            for sentence in sentences
        ]

    def _calc_sentence_similarity(self, sentence_one, sentence_two):

        list_words_one = [token for token in self._nlp(sentence_one)]
        
        list_words_two = [token for token in self._nlp(sentence_two)]
    
        list_all_words = {word: i for i, word in enumerate(set(list_words_one + list_words_two))}
    
        counter_one = Counter(list_words_one)
        counter_two = Counter(list_words_two)
    
        array_one = np.array([counter_one.get(word, 0) for word in list_all_words], dtype=int)
        array_two = np.array([counter_two.get(word, 0) for word in list_all_words], dtype=int)
    
        return 1 - nltk.cluster.util.cosine_distance(array_one, array_two)        

    def _calc_similarity_matrix(self, sentences):
    
        num_sentences = len(sentences)
    
        similarity_matrix = np.zeros((num_sentences, num_sentences))

        for i, sentence_one in enumerate(sentences):
        
            for j in range(i + 1, num_sentences):
            
                similarity = self._calc_sentence_similarity(sentence_one, sentences[j])
            
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _calc_pagerank(self, similarity_matrix):

        similarity_graph = nx.from_numpy_array(similarity_matrix)
        
        scores = nx.pagerank(similarity_graph)

        return scores

    def summarize(self, text):

        self.list_sentences = self._tokenize(text)

        list_preprocessed_sentences = self._preprocess_sentences(self.list_sentences)

        similarity_matrix = self._calc_similarity_matrix(list_preprocessed_sentences)

        self.scores = self._calc_pagerank(similarity_matrix)
        
        self.sorted_scores = [(self.scores[i], sentence) for i, sentence in enumerate(self.list_sentences)]
    
        list_best_sentences = [sentence for _, sentence in heapq.nlargest(self.n_sentences, self.sorted_scores)]
        
        return list_best_sentences