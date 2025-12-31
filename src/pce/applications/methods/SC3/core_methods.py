import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
from .core_functions import calculate_distance, transformation, consensus_matrix, estkTW
from .biology import get_de_genes, get_marker_genes, get_outl_cells

class SC3:
    """
    Python implementation of the SC3-Nature methods-2017 algorithm.
    Mirrors the logic of the R SC3-Nature methods-2017 package.
    """
    
    def __init__(self, data, gene_filter=True, pct_dropout_min=10, pct_dropout_max=90, 
                 d_region_min=0.04, d_region_max=0.07, svm_max=5000, 
                 svm_num_cells=None,
                 n_cores=None, seed=None):
        """
        Initialize SC3-Nature methods-2017 object.
        
        Parameters
        ----------
        data : np.ndarray
            Input data matrix (n_cells, n_genes).
        gene_filter : bool
            Whether to perform gene filtering.
        pct_dropout_min : int
            Minimum dropout percentage (inclusive lower bound check uses >).
        pct_dropout_max : int
            Maximum dropout percentage (inclusive upper bound check uses <).
        """
        # 1. Gene Filtering (Crucial Step)
        self.original_n_genes = data.shape[1]
        self.gene_mask = None
        
        if gene_filter:
            # R: dropouts <- rowSums(counts(object) == 0)/ncol(object)*100
            # Python data is (cells, genes), so we want colSums (axis=0) / n_cells
            n_cells = data.shape[0]
            dropouts = np.sum(data == 0, axis=0) / n_cells * 100
            
            # R: f_data$sc3_gene_filter <- dropouts < pct_dropout_max & dropouts > pct_dropout_min
            mask = (dropouts < pct_dropout_max) & (dropouts > pct_dropout_min)
            
            self.data = data[:, mask]
            self.gene_mask = mask
            
            n_kept = np.sum(mask)
            if n_kept == 0:
                warnings.warn("All genes were removed after gene filter! Reverting to full dataset.")
                self.data = data
                self.gene_mask = np.ones(self.original_n_genes, dtype=bool)
            elif n_kept < self.original_n_genes:
                # print(f"Gene filter: kept {n_kept} of {self.original_n_genes} genes.")
                pass
        else:
            self.data = data
            
        self.n_cells, self.n_genes = self.data.shape
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Parameters
        self.d_region_min = d_region_min
        self.d_region_max = d_region_max
        self.svm_max = svm_max
        
        # SVM Subsampling Logic (Happens AFTER gene filtering)
        self.svm_train_inds = None
        self.svm_study_inds = None
        
        if self.n_cells > self.svm_max or svm_num_cells is not None:
            # Determine number of training cells
            n_train = svm_num_cells if svm_num_cells is not None else self.svm_max
            if n_train > self.n_cells:
                n_train = self.n_cells
            
            # Sample indices
            all_inds = np.arange(self.n_cells)
            self.svm_train_inds = sorted(self.rng.choice(all_inds, n_train, replace=False))
            self.svm_study_inds = sorted(list(set(all_inds) - set(self.svm_train_inds)))
            
            self.train_data = self.data[self.svm_train_inds, :]
            if len(self.svm_study_inds) > 0:
                self.study_data = self.data[self.svm_study_inds, :]
            else:
                self.study_data = None
        else:
            self.train_data = self.data
            self.study_data = None
            
        # State storage
        self.distances = {}
        self.transformations = {}
        self.kmeans_partitions = []
        self.consensus_matrix = None
        self.labels = None
        self.biology = {}
        
        # Dimensions for K-means (n_dim)
        # Calculate based on training set size
        n_train_cells = self.train_data.shape[0]
        min_dim = int(np.floor(d_region_min * n_train_cells))
        max_dim = int(np.ceil(d_region_max * n_train_cells))
        min_dim = max(1, min_dim)
        max_dim = max(min_dim, max_dim)
        
        self.n_dims = list(range(min_dim, max_dim + 1))
        
        if len(self.n_dims) > 15:
            self.n_dims = sorted(self.rng.choice(self.n_dims, 15, replace=False))
            
    def estimate_k(self):
        """
        Estimate optimal k using Tracy-Widom theory.
        Runs on full dataset as per R implementation.
        """
        return estkTW(self.data)
        
    def calc_dists(self):
        """
        Calculate distance matrices on TRAINING data.
        """
        metrics = ['euclidean', 'pearson', 'spearman']
        for metric in metrics:
            try:
                self.distances[metric] = calculate_distance(self.train_data, metric)
            except Exception as e:
                warnings.warn(f"Failed to calculate {metric} distance: {e}")
                
    def calc_transfs(self):
        """
        Calculate transformations on TRAINING data.
        """
        if not self.distances:
            self.calc_dists()
            
        trans_methods = ['pca', 'laplacian']
        
        for dist_name, dist_mat in self.distances.items():
            for trans_method in trans_methods:
                key = f"{dist_name}_{trans_method}"
                try:
                    self.transformations[key] = transformation(dist_mat, trans_method)
                except Exception as e:
                    warnings.warn(f"Failed to calculate transformation {key}: {e}")

    def kmeans(self, n_clusters, n_init=10, max_iter=300):
        """
        Run K-means on transformations of TRAINING data.
        """
        if not self.transformations:
            self.calc_transfs()
            
        self.kmeans_partitions = []
        
        for trans_key, trans_mat in self.transformations.items():
            for d in self.n_dims:
                if d > trans_mat.shape[1]:
                    continue
                
                X_subset = trans_mat[:, :d]
                
                try:
                    # Note: n_cores/n_jobs removed from KMeans as it is deprecated in newer sklearn
                    km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, 
                                random_state=self.rng)
                    km.fit(X_subset)
                    self.kmeans_partitions.append(km.labels_)
                except Exception as e:
                    pass
                    
        self.kmeans_partitions = np.array(self.kmeans_partitions).T 
        
    def consensus(self, n_clusters):
        """
        Calculate consensus matrix and clustering on TRAINING data.
        """
        if len(self.kmeans_partitions) == 0:
            raise RuntimeError("No K-means partitions generated.")
            
        self.consensus_matrix = consensus_matrix(self.kmeans_partitions)
        
        cons_dists = pdist(self.consensus_matrix, metric='euclidean')
        Z = linkage(cons_dists, method='complete')
        cutree_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        # Re-index clusters to match dendrogram order (matches R's reindex_clusters)
        # 1. Get leaf order from dendrogram
        leaves_order = leaves_list(Z)
        
        # 2. Sort labels by their appearance in the dendrogram
        ordered_labels = cutree_labels[leaves_order]
        
        # 3. Find unique labels in order of appearance
        unique_labels_in_order = []
        seen = set()
        for lbl in ordered_labels:
            if lbl not in seen:
                unique_labels_in_order.append(lbl)
                seen.add(lbl)
        
        # 4. Map old_label -> new_label (1..k)
        # unique_labels_in_order[0] -> 1
        # unique_labels_in_order[1] -> 2
        mapping = {old: new for new, old in enumerate(unique_labels_in_order, 1)}
        
        new_labels = np.array([mapping[l] for l in cutree_labels])
        
        # Return 0-based for Python consistency
        return new_labels - 1

    def run_svm(self, train_labels):
        """
        Train SVM on training cells and predict study cells.
        """
        if self.study_data is None:
            return train_labels
            
        clf = SVC(kernel='linear')
        clf.fit(self.train_data, train_labels)
        study_labels = clf.predict(self.study_data)
        
        # Merge labels
        full_labels = np.zeros(self.n_cells, dtype=int)
        full_labels[self.svm_train_inds] = train_labels
        full_labels[self.svm_study_inds] = study_labels
        
        return full_labels

    def calc_biology(self):
        """
        Calculate biological features using FULL dataset (filtered) and FULL labels.
        """
        if self.labels is None:
            raise RuntimeError("Run clustering first.")
            
        self.biology['de'] = get_de_genes(self.data, self.labels)
        self.biology['marker'] = get_marker_genes(self.data, self.labels)
        self.biology['outl'] = get_outl_cells(self.data, self.labels)

    def run(self, n_clusters=None, biology=False, kmeans_nstart=10, kmeans_iter_max=300):
        """
        Run the SC3-Nature methods-2017 pipeline (with optional SVM hybrid mode).
        """
        if n_clusters is None:
            n_clusters = self.estimate_k()
            print(f"Estimated k: {n_clusters}")
            
        self.calc_dists()
        self.calc_transfs()
        self.kmeans(n_clusters=n_clusters, n_init=kmeans_nstart, max_iter=kmeans_iter_max)
        
        # Clustering on training data
        train_labels = self.consensus(n_clusters=n_clusters)
        
        if self.svm_train_inds is not None:
            # Predict rest
            self.labels = self.run_svm(train_labels)
        else:
            self.labels = train_labels
            
        if biology:
            self.calc_biology()
            
        return self.labels, self.biology