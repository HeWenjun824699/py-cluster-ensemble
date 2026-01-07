import numpy as np
import warnings
from sklearn.svm import SVC
from .core_functions import estkTW
from .biology import get_de_genes, get_marker_genes, get_outl_cells
from ....generators.sc3_generator import sc3_generator
from ....consensus.sc3 import sc3

class SC3:
    """
    Python implementation of the SC3-Nature methods-2017 algorithm.
    Mirrors the logic of the R SC3-Nature methods-2017 package.
    """
    
    def __init__(self, data, gene_filter=True, pct_dropout_min=10, pct_dropout_max=90, 
                 d_region_min=0.04, d_region_max=0.07, svm_max=5000, 
                 svm_num_cells=None, n_cores=None, seed=None):
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
        # 1. Estimate K if needed
        if n_clusters is None:
            n_clusters = self.estimate_k()
            print(f"Estimated k: {n_clusters}")

        # 2. Generate Base Partitions (Using sc3_generator)
        print("Generating base partitions...")
        bps_list = []
        # Triple loop: Metric -> Transformation -> Dimensions
        metrics = ['euclidean', 'pearson', 'spearman']
        trans_methods = ['pca', 'laplacian']
        for metric in metrics:
            for trans in trans_methods:
                for d in self.n_dims:
                    # Call generator for exactly 1 partition with specific params
                    # Pass master seed but ideally SC3 doesn't vary much here except kmeans init
                    bp = sc3_generator(
                        X=self.train_data,
                        nClusters=int(n_clusters),
                        nPartitions=1,  # Generate 1 specific column
                        metric=metric,  # Fixed metric
                        trans_method=trans,  # Fixed transformation
                        n_eigen=d,  # Fixed dimension
                        seed=self.seed if self.seed else 2026,
                        maxiter=kmeans_iter_max,
                        n_init=kmeans_nstart
                    )
                    bps_list.append(bp)

        # Stack all columns: (n_samples, n_total_combinations)
        BPs = np.hstack(bps_list)
        actual_n_partitions = BPs.shape[1]
        print(f"Generated {actual_n_partitions} base partitions.")

        # 3. Consensus Clustering (Using sc3_consensus)
        # Standard SC3 workflow typically uses all generated base partitions and runs only once
        nBase = len(metrics) * len(trans_methods) * len(self.n_dims)
        print("Computing consensus...")
        labels_list, _, self.consensus_matrix = sc3(
            BPs=BPs,
            nClusters=n_clusters,
            nBase=nBase,
            nRepeat=1,
            seed=self.seed if self.seed else 2026,
            return_matrix=True
        )

        # sc3_consensus returns a list, take the result of the first run
        train_labels = labels_list[0]

        # 4. SVM Hybrid Extension (if applicable)
        if self.svm_train_inds is not None:
            print("Running SVM extension...")
            self.labels = self.run_svm(train_labels)
        else:
            self.labels = train_labels

        # 5. Biological Analysis (Optional)
        if biology:
            print("Calculating biological features...")
            self.calc_biology()

        return self.labels, self.biology
