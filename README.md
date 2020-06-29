# OCKRA

[OCKRA](https://www.mdpi.com/1424-8220/16/10/1619)<sup>1</sup> (One-Class K-means with Randomly-projected features Algorithm) is an ensemble of one-class classiﬁers, which are based on multiple projections of the dataset according to random subsets of features.

OCKRA has been designed with the aim of improving the detection performance in the problem posed by the [PRIDE](https://www.sciencedirect.com/science/article/pii/S002002551630576X)<sup>2</sup> (Personal RIsk DEtection) dataset.


## Usage

A practical example about how to use OCKRA can be seen in [Example](https://github.com/Miel15/OCKRA-classifier/tree/master/Example).


## Requirements

Python 3, Scikit-learn, Pandas, Numpy


## User Guide

```
class OCKRA(classifier_count=100, K=10, use_dist_threshold=False, user_threshold=95)
```

### Parameters

 - **classifier_count : int, default=100**
Number of classifiers in the ensemble.

- **k : int, default=10**
The number of clusters to form as well as the number of centroids to generate.

- **use_dist_threshold : bool, default=False**
Computes the distance threshold.

### Methods

```
fit(self, X, y)
```
Training of the OCC algorithm.

**Parameters**

 - **X: *array-like* of shape (n_samples, n_features)**
Set of samples, where *n_samples* is the number of instances and *n_features* is the number of features. 
- **y: *Ignored* array-like of shape (n_samples, 1)**
Not used, present for API consistency by convention.

**Return**
- **self: *object*** 

```
predict(self, X)
```
Perform classification on samples in *X*. For a one-class model, +1 or -1 is returned.

**Parameters**

 - **X: *array-like* of shape (n_samples, n_features)**
Set of samples, where *n_samples* is the number of instances and *n_features* is the number of features.

**Return**
- **y_pred: *ndarray* of shape (n_samples, )**
Class labels for samples in *X*.


```
score_samples(self, X)
```
Raw scoring function of the samples. Compute the similarity value.

**Parameters**

 - **X: *array-like* of shape (n_samples, n_features)**
The data array.

**Return**
- **y_pred: *ndarray* of shape (n_samples, )**
Class labels for samples in *X*.


## Algorithm

### Training phase
   * **Input:**
        * *T*: training dataset;
        * *N*: number of classifiers in the ensemble;
        * *k*: clusters obtained using k-means++.
   * **Local Variables:**
        * *SelectedFeatures*: randomly-selected features;
        * *T'*: projected dataset *T* over the randomly-selected features;
        * *Centres*: centers of the *k* clusters obtained using k-means++;
        * *δ<sub>i</sub>*:  average distance between all objects in *T'*.
    * **Output:**
        * *P*: the set of classifiers parameters (randomly-selected features, the centroids of each cluster and the distance threshold).
    * **Start**:
        1. Set *P* initially empty; i.e., *P* ← {}
        2. **for** *i*= 1..*N* **do**:
	        1. *SelectedFeatures*  ← RandomFeatures(T)
	        2. *T'* ← Project(*T*, *SelectedFeatures*)
	        3. *Centres* ← ApplyKMeansAndComputeCentres(*T′*)
	        4. *δ<sub>i</sub>* ← AvgDistances(*T′*)
	        5. *P* ← *P* U { (*SelectedFeatures*, *Centres*, *δ<sub>i</sub>* ) }
	        6. *P*←*P*U{(*T<sub>i</sub>*, *δ<sub>i</sub>* )}
        3. **end for**
        4. **return** *P*


### Classification phase
   * **Input:**
        * *O*: object to be classified;
        * *P*: the set of parameters computed in the training phase.
   * **Local Variables:**
        * *O'*: projected object *O* over the randomly-selected features;
        * *d<sub>min</sub>*:  the nearest cluster to *O′* (smallest Euclidean distance between the selecting centroid and the object *O'*).
    * **Output:**
        * *s*: similarity value (zero indicates an anormal behaviour and one represents normal behavior).
    * **Start:**
        1. Let *s* ← 0 be the similarity value computed by the classifiers
        2. **for each** (*Features<sub>i</sub>*, *Centres<sub>i</sub>*, *δ<sub>i</sub>*) **in** *P* **do**:
            1. *O'* ← Project(*O*, *Features<sub>i</sub>*)
            2. *d<sub>min</sub>* ← min<sub>c<sub>j</sub> in Centres<sub>i</sub></sub> (EuclideanDistance(O', *c<sub>j</sub>*))
            3. *s* ← *s* + *e*^(-0.5(*d<sub>min</sub>* ∕ *δ<sub>i</sub>* )^2 )
        3. **end for**
        4. **return** *s* / | *P* |


## References

 1. *For more information about OCKRA, please read:*
	 J. Rodríguez, A. Barrera-Animas, L. Trejo, M. Medina-Pérez and R. Monroy, ["Ensemble of one-class classifiers for personal risk detection based on wearable sensor data"](https://www.mdpi.com/1424-8220/16/10/1619), _Sensors_, vol. 16, no. 10, pp. 1619, Sep. 2016.
	 
 2. *For more information about the PRIDE dataset, please read:*
	A. Y. Barrera-Animas, L. A. Trejo, M. A. Medina-Pérez, R. Monroy, J. B. Camiña and F. Godínez, ["Online personal risk detection based on behavioural and physiological patterns"](https://www.sciencedirect.com/science/article/pii/S002002551630576X), _Information Sciences_, vol. 384, pp. 281-297, Apr. 2017.
