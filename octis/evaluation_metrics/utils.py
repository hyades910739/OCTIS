from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np


class KeyedVectorsMixin:
    "A Mixin to load or process KeyedVectors."
    default_model_name = 'word2vec-google-news-300'

    def load_keyedvectors(self, model, model_name):
        """
        Load KeyedVectors

        Parameters
        ----------
        :model : Either None, a KeyedVectors instance, or a dict with key is word (str) and value is
        word embedding (1d numpy array).
        :model_name : A string specify the pre-train embedding name to load. Only used when model is None.
        """
        if model is not None:
            if isinstance(model, KeyedVectors):
                pass
            elif isinstance(model, dict):
                model = KeyedVectorsMixin._convert_dict_to_keyedvectors(model)
            else:
                raise ValueError('model should be either KeyedVectors or a dictionary.')
        elif model_name is not None:
            model = api.load(model_name)
        else:
            model = api.load(self.default_model_name)
        self.wv = model

    @staticmethod
    def _convert_dict_to_keyedvectors(dic):
        vector_size = dic.values()
        arr = next(iter(dic.values()))
        assert isinstance(arr, np.ndarray)
        assert len(arr.shape) == 1
        vector_size = arr.shape[0]
        kv = KeyedVectors(vector_size)
        kv.key_to_index = {k: idx for idx, k in enumerate(dic.keys())}
        kv.vectors = np.stack(list(dic.values()))
        return kv
