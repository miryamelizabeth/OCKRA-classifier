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
  
