import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from .core_functions import calculate_distance, transformation, consensus_matrix, estkTW
from .biology import get_de_genes, get_marker_genes, get_outl_cells

class SC3:
    """
    Python implementation of the SC3 algorithm.
    Mirrors the logic of the R SC3 package.
    """
    
    def __init__(self, data, gene_filter=True, pct_dropout_min=10, pct_dropout_max=90, 
                 d_region_min=0.04, d_region_max=0.07, svm_max=5000, 
                 svm_num_cells=None,
                 n_cores=None, seed=None):
        """
        Initialize SC3 object.
        
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
        # Changed to dictionaries to support multiple k
        self.kmeans_partitions = {} # k -> np.ndarray
        self.consensus_matrices = {} # k -> np.ndarray
        self.labels = {} # k -> np.ndarray
        self.biology = {} # k -> dict
        
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

    def kmeans(self, n_clusters_list, n_init=10, max_iter=300):
        """
        Run K-means on transformations of TRAINING data for multiple k.
        """
        if not self.transformations:
            self.calc_transfs()
            
        # Reset partitions
        self.kmeans_partitions = {}
        
        # Ensure input is a list
        if isinstance(n_clusters_list, int):
            n_clusters_list = [n_clusters_list]
        
        for k in n_clusters_list:
            partitions_k = []
            
            for trans_key, trans_mat in self.transformations.items():
                for d in self.n_dims:
                    if d > trans_mat.shape[1]:
                        continue
                    
                    X_subset = trans_mat[:, :d]
                    
                    try:
                        # Note: n_cores/n_jobs removed from KMeans as it is deprecated in newer sklearn
                        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, 
                                    random_state=self.rng)
                        km.fit(X_subset)
                        partitions_k.append(km.labels_)
                    except Exception as e:
                        pass
            
            if partitions_k:
                self.kmeans_partitions[k] = np.array(partitions_k).T
            else:
                warnings.warn(f"No K-means partitions generated for k={k}")

    def consensus(self):
        """
        Calculate consensus matrix and clustering on TRAINING data for all k.
        """
        if not self.kmeans_partitions:
            raise RuntimeError("No K-means partitions generated.")
            
        self.consensus_matrices = {}
        self.labels = {}
        
        for k, partitions in self.kmeans_partitions.items():
            consensus_mat = consensus_matrix(partitions)
            self.consensus_matrices[k] = consensus_mat
            
            # Hierarchical clustering on consensus matrix
            cons_dists = pdist(consensus_mat, metric='euclidean')
            
            # Handle edge case where consensus matrix is perfect (distance 0)
            # or errors in linkage
            try:
                Z = linkage(cons_dists, method='complete')
                labels = fcluster(Z, t=k, criterion='maxclust')
                # 0-based
                self.labels[k] = labels - 1
            except Exception as e:
                warnings.warn(f"Consensus clustering failed for k={k}: {e}")

    def run_svm(self):
        """
        Train SVM on training cells and predict study cells for all k.
        """
        if self.study_data is None:
            return
            
        # We need to update labels for each k
        # Current self.labels contains training labels
        
        for k in list(self.labels.keys()):
            train_labels = self.labels[k]
            
            clf = SVC(kernel='linear')
            clf.fit(self.train_data, train_labels)
            study_labels = clf.predict(self.study_data)
            
            # Merge labels
            full_labels = np.zeros(self.n_cells, dtype=int)
            full_labels[self.svm_train_inds] = train_labels
            full_labels[self.svm_study_inds] = study_labels
            
            self.labels[k] = full_labels

    def calc_biology(self):
        """
        Calculate biological features using FULL dataset (filtered) and FULL labels for all k.
        """
        if not self.labels:
            raise RuntimeError("Run clustering first.")
            
        self.biology = {}
        
        for k, labels in self.labels.items():
            self.biology[k] = {}
            self.biology[k]['de'] = get_de_genes(self.data, labels)
            self.biology[k]['marker'] = get_marker_genes(self.data, labels)
            self.biology[k]['outl'] = get_outl_cells(self.data, labels)

    def run(self, n_clusters=None, biology=False):
        """
        Run the SC3 pipeline (with optional SVM hybrid mode).
        
        Parameters
        ----------
        n_clusters : int or list of int, optional
            Number of clusters k. If None, estimated automatically.
        biology : bool
        
        Returns
        -------
        tuple
            (labels_dict, biology_dict)
            Where keys are k.
        """
        if n_clusters is None:
            estimated_k = self.estimate_k()
            print(f"Estimated k: {estimated_k}")
            n_clusters = [estimated_k]
        elif isinstance(n_clusters, int):
            n_clusters = [n_clusters]
            
        self.calc_dists()
        self.calc_transfs()
        self.kmeans(n_clusters_list=n_clusters)
        self.consensus()
        
        # If SVM is active, extend labels to full dataset
        if self.svm_train_inds is not None:
            self.run_svm()
            
        if biology:
            self.calc_biology()
            
        return self.labels, self.biology
