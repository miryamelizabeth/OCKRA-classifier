# Example created by Miryam Elizabeth Villa-Pérez


import pandas as pd
import ockra

from sklearn.metrics import roc_auc_score


def load_datasets(train_file, test_file):
        
    print('Reading datasets...')
    
    train = pd.read_csv(train_file, header=None)
    test = pd.read_csv(test_file, header=None)
        
    return train, test


# Reading files
train, test = load_datasets('training.csv', 'testing.csv')


# Split train, test
rows, columns = train.shape

X_train = train.iloc[:, :columns-1]
y_train = train.values[:, -1]

X_test = test.iloc[:, :columns-1]
y_test = test.values[:, -1]


# Training phase
# ---------------------------
# We use by default the parameters of the article:
# 'Ensemble of One-Class Classifiers for Personal Risk Detection Based on Wearable Sensor Data' (https://doi.org/10.3390/s16101619)
# classifier_count = 100, K = 10
# ---------------------------
# If we want other parameters...
# OCKRA(classifier_count=50, K=5, use_dist_threshold=True)

print('Training classifier...')
clf = ockra.OCKRA()
clf.fit(X_train, y_train)


# Testing phase
print('Classifier trained!')
print('Testing classifier...')
        
y_pred = clf.score_samples(X_test)
auc = roc_auc_score(y_test,  y_pred)
print(f'Testing AUC: {auc if auc > .5 else 1 - auc}')
