import time
from scipy.stats import mode
from sklearn.cluster import KMeans
import pickle as pkl
import numpy as np


class FaceVerifier(object):

    def __init__(self):
        start_time = time.time()

        filename = 'models/kmeans_model.sav'
        self.kmeans_model = pkl.load(open(filename, 'rb'))

        filename = 'models/svc_model.sav'
        self.svc_model = pkl.load(open(filename, 'rb'))

                #    classifier_model = pickle.load(open(filename, 'rb'))
        self.kmean_classes = None
        with open("models/kmeans_classes.txt", "rb") as fp:   # Unpickling
            self.kmean_classes = pkl.load(fp)

        self.svc_classes = np.load('models/svc_classes.npy')

        end_time = time.time() - start_time
        print('Time for loading classifier: {}'.format(end_time))


    def getPredictions(self, embedding, enable_kmeans=True):

        overall_time = time.time()
        samples = embedding[0]
        
        svc_yhat_class = self.svc_model.predict(embedding)
        svc_yhat_prob = self.svc_model.predict_proba(embedding)

        yhat_class, yhat_prob = self._sim_predict(self.kmeans_model, samples, self.kmean_classes, top_n=5)

        class_probability = yhat_prob[0] * 100
        predict_names = yhat_class
        #print('Predicted: %s (%.3f)' % (predict_names, class_probability))
        compute_time = time.time() - overall_time

        res = {'svc_class': self.svc_classes[svc_yhat_class[0]], 'confidence': svc_yhat_prob[0, svc_yhat_class[0]]*100}

        if enable_kmeans:
            res.update({'predictions': predict_names, 'probs': list(yhat_prob), 'compute_time': compute_time})
        

        return res


    def _sim_predict(self, model, new_img_f, kmean_classes, top_n=1, n_classes=5):
        
        # cluster labels do not match with actual order of train data. so find indices to reorder cluster centres
        kmeans = model
        steps = np.linspace(0, len(kmeans.labels_), num=n_classes+1)
        orig_labels = []
        last_val = 0
        for i in steps[1:]:
            cluster_labels = kmeans.labels_[last_val:int(i)]
            last_val = int(i)
            orig_labels += [mode(cluster_labels)[0][0]]

        # new_map = {}
        # for i, label in enumerate(encode_labels):
        #     new_map[]


        relabeled = kmeans.cluster_centers_[orig_labels]
        sims = np.array([])
        for i in range(relabeled.shape[0]):
            sim = np.dot(relabeled[i],new_img_f)
            sims = np.append(sims,sim)
        sims_top_n = sims.argsort()[-top_n:][::-1]
        classes = sims_top_n

        classes = [kmean_classes[val] for val in classes]
        
        #print(f'new_classes: {classes}')
        probs = sims[sims_top_n]
        #print(f'classes: {classes}')
        return classes, probs



        