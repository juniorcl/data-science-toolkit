import nltk
import spacy
import heapq

from collections import defaultdict


class WordFreqSummarizer():

    def __init__(self, n_sentences, lemma=True, spacy_model="pt_core_news_sm"):

        self.n_sentences = n_sentences
        
        self._nlp = spacy.load(spacy_model)
        
        self._lemma = lemma

    def _tokenize(self, text):
        
        return nltk.sent_tokenize(text)
        
    def _preprocess_text(self, sentences):
            
        preprocessed_sent = [
            " ".join(
                token.lemma_ if self._lemma else token.text
                for token in self._nlp(sentence.lower())
                if not token.is_stop and not token.is_punct
            )
            for sentence in sentences
        ]

        return " ".join(preprocessed_sent)

    def _get_word_frequency(self, text):

        words = nltk.word_tokenize(text)

        word_freq = nltk.FreqDist(words)

        word_freq_max = max(word_freq.values())

        return {word: value / word_freq_max for word, value in word_freq.items()}

    def _get_sent_scores(self, word_freqs, sentences):

        sent_scores = defaultdict(int)

        for sent in sentences:
            for word in nltk.word_tokenize(sent):
                if word in word_freqs:
                    sent_scores[sent] += word_freqs[word]
        
        return sent_scores

    def _get_best_n_sentences(self):

        return heapq.nlargest(self.n_sentences, self.sent_scores, key=self.sent_scores.get)
        
    def summarize(self, text):

        self.sentences = self._tokenize(text)

        preprocessed_text = self._preprocess_text(self.sentences)

        word_freq_dict = self._get_word_frequency(preprocessed_text)

        self.sent_scores = self._get_sent_scores(word_freq_dict, self.sentences)

        best_sents = self._get_best_n_sentences()
        
        return best_sents
    

