import torch
from src.get_data import get_mnist

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

(X, _), (X_test, _) = get_mnist('./data', 
                                sample_size=5, 
                                sample_size_test=5,
                                batch_size=1, 
                                seed=1, 
                                device=device,
                                classes=None)
size = X.shape
flattened_size = size[-1]*size[-2]*size[-3]
X = X.reshape(-1, flattened_size)
print(X.shape)