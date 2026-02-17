# Spectral-Grounding-Specialized-MLP-for-extreme-noise-robustness
A Custom specialized MLP Designed to handle noise with a very consistent Accuracy on 1000+ Samples. up to 95% consistent Accuracy on 10 trials (10 different initialization), each trial consist of 900 epoch trainings. Using a Highly Specialized Custom Module Called "Abstract Weight Encoder", or short as AWE.


# MLP Introduction
Multilayer Perceptron (MLP) is a foundational, supervised feed-forward artificial neural network consisting of at least three layers (input, hidden, output) of fully connected neurons. It uses nonlinear activation functions (like ReLU or sigmoid) and backpropagation to learn complex, non-linearly separable relationships, commonly used for classification and regression tasks. 

^. MLP and Setup Requirements:
~ Numpy and sklearn library
~ 16 hidden dimension
~ Input dim and output dim depends on sklearn samples (Mostly 1000-10000 samples)

^. Experiment Note:
The MLP i used is mostly small, To further test the capabilites of the abstract weight encoder in smaller Datasets and fewer Parameters, Parameters can be scaled, and it doesn't cause Accuracy degradation, with a very consistent results, especially On linear Make_classification module with accuracy being as consistent as much as 90-95% Accuracy with:
class separation = 1.5
random_state = 99



# Specific Math Used
1. Eigenvalue:
   The eigenvalue equation in machine learning is 
Av = Bv, where a square matrix A
 (e.g., covariance matrix) acts on an eigenvector v, resulting in a scaled version of itself by the scalar eigenvalue 
B, It is fundamental for dimensionality reduction (PCA), spectral clustering, and SVD to identify principal directions of data variance.
   Eigenvalue has An important Role in AWE, supporting to calculate the implicit structure given a covariance of a single or multi batch matrix, A neccessary equation to keep both input and output to have the same implicit eigenvalue energy necessary for capturing valuable covariance geometric structure despite noise dominance inside a given matrix. 
Code form:
```
        eps = 1e-5
        mag = np.mean(np.linalg.norm(x, axis=1))
    
        gradient = np.gradient(x)
        val = [np.linalg.norm(g) for g in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps

        structured_noise = np.random.uniform(0, mag, size=(len(x), len(x[0])))
        X = np.vstack((x, structured_noise))
        X_centered = X - X.mean(axis=0)
        cov = np.cov(X_centered, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        
        eigenvalues = eigenvalues[idx]
        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        k = np.searchsorted(energy, 0.90) + 1
```

in where K is a product of eigenvalue energy after binary search, in which the necessary for efficient Categorization, and allows for mapping continous eigenvalue outputs to discrete bins, in technical term, necessary for calculating an efficient index in which a necessary eigenvalue maintains a stable energy in which order of covariance, categorized to int scalar to efficiently be used for eigenvalue ratio for later equations.


# Instructions:
To use AWE You must download or import Python library such as:
1. Numpy
2. Sklearn
Note: Supports python 3.14+

^. Step By Step usage:
~ download my AWE Encoder, Plug it in python environment along with Any Numpy MLP SetUp
~ My current MLP Already has the single weight (self.W) plugged with special_weight.weight_encoder(), or you can manually test or add more weights
~ Create and import make_classfication() to directly test and dont forget to import train_test_split() too.
~ you're ready to try the weight encoder and see the consistent accuracy. 


# Test Results of my Experiement:
for a convincing results, the data for training i used is train_test_split() in which X, and y, which correlates for input and correct training data set, and random_state is 99.
^. 1. Make_classification:
   Code form:
```
X, y_raw = make_classification(
    n_samples=1000,
    n_features=3,
    n_classes=3,
    n_informative=3,
    n_redundant=0,
    class_sep=1.5,
    random_state=99
)
```
   ~ on regular X using:
```
train_test_split(X, y, test_size=0.9, random_state=99)
```
   the underlying accuracy reaches up to 86%-93% on 1000 samples consistent accross  10 trials (10*1000 epochs) with 16 hidden_dim, in which each trial resets the models training to first initialization. and 93-97% accuracy on up to 2000 hidden_dim with 1000 samples consistently accross 10 trials.
   
   ~ When The model was trained using X_noisy, in which X_noisy is a distorted input in which X + noise_scale, where noise_scale is 0.9 (90% noise) using np.random.uniform(0, noise_scale, size=x.shape), and it was put inside 
```
train_test_split(X_noisy, y, test_size=0.9, random_state=99)
```

   the underlying accuracy consistently ranging from 79% - 82% on 1000 samples with 16 hidden_dim.
^. 2. Make_moons:
   code form:
 ```
 X, y_raw = make_moons(
    n_samples = 5000, 
    noise=0.5,  
    random_state=99) 
 ```
With 50% noise,
The underlying Accuracy is around 73-76% from epoch 0 to 900 consistently accross 10 different initialization with 5000 samples and 5000 hidden dim parameters.
   
   
  


