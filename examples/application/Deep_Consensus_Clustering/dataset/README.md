# Dataset Information

## ⚠️ Disclaimer: Mock Data
The dataset provided in this directory consists of **synthetic mock data** generated solely for demonstration and testing purposes. 

**Please Note:**
* This data **does not** represent real-world clinical distributions or biological patterns.
* It is intended to verify the functionality of the code pipeline (e.g., verifying data loading, model training, and clustering flows).
* Do not use this data for clinical interpretation or performance benchmarking against real-world baselines.

## File Structure
The data is stored in binary format using `pickle` to ensure compatibility with the `AKIData` loader class. Each file contains a tuple `(X, Y)`:
* **X**: Feature matrix (n_samples, n_features)
* **Y**: Label/Outcome vector (n_samples,)

Files generated:
* `data.pkl`: The complete dataset.
* `data_train.pkl`: Training set (60%).
* `data_valid.pkl`: Validation set (30%).
* `data_test.pkl`: Test set (10%).

## Data Generation Script
If you wish to modify the dataset size, feature dimensions, or noise levels, you can use the following script to regenerate the mock data.

Save the code below as `generate_mock_data.py` and run it:

```python
import os
import pickle
import numpy as np

def generate_mock_data(n_samples=1000, n_timesteps=14, n_features=26, save_dir='./dataset'):
    """
    Generates synthetic mock data, including the complete 'data.pkl' 
    and pre-split train/test/valid files.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Generating {n_samples} mock samples...")

    # 1. Generate features (X) and labels (y) for all samples
    # Shape: (n_samples, n_timesteps, n_features)
    X = np.random.randn(n_samples, n_timesteps, n_features).astype(np.float32)

    # Simulate Padding: Randomly mask the initial timesteps with 0s
    for i in range(n_samples):
        valid_len = np.random.randint(2, n_timesteps + 1)
        X[i, :-(valid_len), :] = 0.0

    # Labels (n_samples,) flattened to 1D to match BCELoss input requirements
    y = np.random.randint(0, 2, size=(n_samples,)).astype(np.float32)

    # 2. Prepare complete dataset (Tuple)
    data_all = (X, y)

    # 3. Split dataset (Ratio 6:3:1)
    n_train = int(n_samples * 0.6)
    n_test = int(n_samples * 0.3)
    n_valid = n_samples - n_train - n_test  # The remainder goes to the validation set

    # Slicing operations
    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:n_train + n_test]
    y_test = y[n_train:n_train + n_test]

    X_valid = X[n_train + n_test:]
    y_valid = y[n_train + n_test:]

    # Construct Tuples
    data_train = (X_train, y_train)
    data_test = (X_test, y_test)
    data_valid = (X_valid, y_valid)

    # 4. Save all files
    files = {
        'data.pkl': data_all,  # <--- Complete data.pkl added here
        'data_train.pkl': data_train,
        'data_test.pkl': data_test,
        'data_valid.pkl': data_valid
    }

    print(f"Save directory: {save_dir}")
    for fname, data in files.items():
        path = os.path.join(save_dir, fname)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  - Generated: {fname:<15} | Samples: {len(data[0])}")


if __name__ == "__main__":
    generate_mock_data()
```