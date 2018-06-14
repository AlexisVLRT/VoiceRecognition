import FeatureExtractor
from sklearn.neural_network import MLPClassifier
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import time


def generate_sets():
    people = os.listdir('Datasets')
    del people[people.index('Test')]
    X = []
    Y = []
    for person in people:
        samples = os.listdir('Datasets//' + person)
        X += [FeatureExtractor.extract_features('Datasets//{}//{}'.format(person, sample))[0] for sample in samples]
        Y += [int('Alex' in person) for _ in range(len(samples))]

    time.sleep(1)
    X, Y = shuffle(X, Y, random_state=0)
    X = np.array(X)
    return X, Y


if __name__ == '__main__':
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(280, 280), random_state=1)
    X, y = generate_sets()
    model.fit(X, y)
    print(model.predict([FeatureExtractor.extract_features('Datasets//Test//Test1-Alex.wav')[0]]))
