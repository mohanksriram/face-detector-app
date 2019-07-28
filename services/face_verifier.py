import time
from scipy.stats import mode
from sklearn.cluster import KMeans
import pickle as pkl
import numpy as np


class FaceVerifier(object):

    def __init__(self):
        start_time = time.time()

        self.classes = np.load('models/classes.npy')
        self.avg_embeddings = np.load('models/avg_embeddings.npy')
        end_time = time.time() - start_time
        print('Time for loading classifier: {}'.format(end_time))

    def _distance(self, emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

    def _get_most_similar(self, new_embeddings, avg_embeddings):
        sims = []
        for new_embedding in new_embeddings:
            dists = [self._distance(new_embedding, emb2) for emb2 in avg_embeddings]
            ans = (np.argmin(dists), min(dists))
            sims.append(ans)
        return sims

    def getPredictions(self, embedding, enable_kmeans=True):

        overall_time = time.time()
        samples = embedding[0]

        preds = self._get_most_similar(embedding, self.avg_embeddings)

        final_labels = ['stranger']*len(preds)
        for idx, (label, dist) in enumerate(preds):
            if dist < 0.9:
                final_labels[idx] = self.classes[label]

        compute_time = time.time() - overall_time

        res = {'class_name': final_labels[0], 'confidence': 100}
        
        print(f'final_res: {res}')

        return res



        