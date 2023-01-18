from octis.evaluation_metrics.diversity_metrics import WordEmbeddingsInvertedRBO, \
    WordEmbeddingsInvertedRBOCentroid, InvertedRBO
from octis.evaluation_metrics.utils import KeyedVectorsMixin
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cosine
from octis.evaluation_metrics.metrics import AbstractMetric


class WordEmbeddingsRBOMatch(WordEmbeddingsInvertedRBO):
    def __init__(self, weight=0.9, topk=10, normalize=True, model=None, model_name=None):
        """
        Initialize metric WERBO-Match

        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :param weight: Weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
        average overlap. (Default 0.9)
        :param normalize: if true, normalize the cosine similarity
        :model : Either None, a KeyedVectors instance, or a dict with key is word (str) and value is
        word embedding (1d numpy array).
        :model_name : A string specify the pre-train embedding name to load. Only used when model is None.
        """
        super().__init__(normalize=normalize, weight=weight, topk=topk, model=model, model_name=model_name)

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :return WERBO-M
        """
        return 1 - super(WordEmbeddingsRBOMatch, self).score(model_output)


class WordEmbeddingsRBOCentroid(WordEmbeddingsInvertedRBOCentroid):
    def __init__(self, normalize=True, weight=0.9, topk=10, model=None, model_name=None):
        """
        Initialize metric WERBO-Centroid

        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :param weight: Weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
        average overlap. (Default 0.9)
        :param normalize: if true, normalize the cosine similarity
        :model : Either None, a KeyedVectors instance, or a dict with key is word (str) and value is
        word embedding (1d numpy array).
        :model_name : A string specify the pre-train embedding name to load. Only used when model is None.
        """
        super().__init__(normalize=normalize, weight=weight, topk=topk, model=model, model_name=model_name)

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :return WERBO-C
        """
        return 1 - super(WordEmbeddingsRBOCentroid, self).score(model_output)


class WordEmbeddingsPairwiseSimilarity(AbstractMetric, KeyedVectorsMixin):
    def __init__(self, topk=10, model=None, model_name=None):
        """
        Initialize metric WE pairwise similarity

        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :model : Either None, a KeyedVectors instance, or a dict with key is word (str) and value is
        word embedding (1d numpy array).
        :model_name : A string specify the pre-train embedding name to load. Only used when model is None.
        """
        super().__init__()
        self.topk = topk
        self.load_keyedvectors(model, model_name)

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :return WEPS
        """
        topics = model_output['topics']
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            count = 0
            sum_sim = 0
            for list1, list2 in combinations(topics, 2):
                word_counts = 0
                sim = 0
                for word1 in list1[:self.topk]:
                    for word2 in list2[:self.topk]:
                        if word1 in self.wv.key_to_index.keys() and word2 in self.wv.key_to_index.keys():
                            sim = sim + self.wv.similarity(word1, word2)
                            word_counts = word_counts + 1
                sim = sim / word_counts
                sum_sim = sum_sim + sim
                count = count + 1
            return sum_sim / count


class WordEmbeddingsCentroidSimilarity(AbstractMetric, KeyedVectorsMixin):
    def __init__(self, topk=10, model=None, model_name=None):
        """
        Initialize metric WE centroid similarity

        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :model : Either None, a KeyedVectors instance, or a dict with key is word (str) and value is
        word embedding (1d numpy array).
        :model_name : A string specify the pre-train embedding name to load. Only used when model is None.
        """
        super().__init__()
        self.topk = topk
        self.load_keyedvectors(model, model_name)

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :return WECS
        """
        topics = model_output['topics']
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            sim = 0
            count = 0
            for list1, list2 in combinations(topics, 2):
                centroid1 = np.zeros(self.wv.vector_size)
                centroid2 = np.zeros(self.wv.vector_size)
                count1, count2 = 0, 0
                for word1 in list1[:self.topk]:
                    if word1 in self.wv.key_to_index.keys():
                        centroid1 = centroid1 + self.wv[word1]
                        count1 += 1
                for word2 in list2[:self.topk]:
                    if word2 in self.wv.key_to_index.keys():
                        centroid2 = centroid2 + self.wv[word2]
                        count2 += 1
                centroid1 = centroid1 / count1
                centroid2 = centroid2 / count2
                sim = sim + (1 - cosine(centroid1, centroid2))
                count += 1
            return sim / count


def get_word2index(list1, list2):
    words = set(list1)
    words = words.union(set(list2))
    word2index = {w: i for i, w in enumerate(words)}
    return word2index


class WordEmbeddingsWeightedSumSimilarity(AbstractMetric, KeyedVectorsMixin):
    def __init__(self, id2word, topk=10, model=None, model_name=None):
        """
        Initialize metric WE Weighted Sum similarity

        :param id2word: dictionary mapping each id to the word of the vocabulary
        :param topk: top k words on which the topic diversity will be computed
        :model : Either None, a KeyedVectors instance, or a dict with key is word (str) and value is
        word embedding (1d numpy array).
        :model_name : A string specify the pre-train embedding name to load. Only used when model is None.
        """
        super().__init__()
        self.topk = topk
        self.id2word = id2word
        self.load_keyedvectors(model, model_name)

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :return WESS
        """
        beta = model_output['topic-word-distribution']

        wess = 0
        count = 0
        for i, j in combinations(range(len(beta)), 2):
            centroid1 = np.zeros(self.wv.vector_size)
            weights = 0
            for id_beta, w in enumerate(beta[i]):
                centroid1 = centroid1 + self.wv[self.id2word[id_beta]] * w
                weights += w
            centroid1 = centroid1 / weights
            centroid2 = np.zeros(self.wv.vector_size)
            weights = 0
            for id_beta, w in enumerate(beta[i]):
                centroid2 = centroid2 + self.wv[self.id2word[id_beta]] * w
                weights += w
            centroid2 = centroid2 / weights
            wess += cosine(centroid1, centroid2)
        return wess / count


class RBO(InvertedRBO):
    def __init__(self, weight=0.9, topk=10):
        """
        Initialize metric Ranked-biased Overlap

        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed
        :param weight: Weight of each agreement at depth d. When set to 1.0, there is no weight, the rbo returns to
        average overlap. (Default 0.9)
        """
        super().__init__(weight=weight, topk=topk)

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :return RBO
        """
        return 1 - super(RBO, self).score(model_output)


class PairwiseJaccardSimilarity(AbstractMetric):
    def __init__(self, topk=10):
        """
        Initialize metric Pairwise Jaccard Similarity

        Parameters
        ----------
        :param topk: top k words on which the topic diversity will be computed

        """
        super().__init__()
        self.topk = topk

    def score(self, model_output):
        """
        Retrieves the score of the metric

        :return PJS
        """
        topics = model_output['topics']
        sim = 0
        count = 0
        for list1, list2 in combinations(topics, 2):
            intersection = len(list(set(list1[:self.topk]).intersection(list2[:self.topk])))
            union = (len(list1[:self.topk]) + len(list2[:self.topk])) - intersection
            count = count + 1
            sim = sim + (float(intersection) / union)
        return sim / count
