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

2. AME Equations (Abstract Modelling Error):
   AME is a fundamental equations needed for calculating further Abstraction error given the magnitude and the gradient of the input, further modelling a neccessary error given the complexity of input samples. its a necessary component given to calculate a subtract of 1.0 from AME, given AME Range 0 -> 1.0.
   Code Form:
```
    def AME_Encoder(self, x):
        X = np.asarray(x)

        gradient = np.gradient(x)
        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME =  np.log1p(X_mag) * np.log1p(grad_energy) 
        return AME
```
Explanation: the log(x_mag + 1) Provides a log value by a non-polynomial function of X_mag in order to express a finite sum of terms consisting of constants and variables raised to whole Number exponents to help identify the logarithmic scale of X_mag. and multiplcation with log(1 + grad_energy) to normalize the range of AME to 0 -> 1, given the positive value of each variables.

in Which High AME, AME > 0.75. Correlates towards such High Error indication of Possible Ongoing Abstraction due to the Complexity of The input samples. In Which Low AME Correlates towards More Efficient Abstraction and Low Possible Error can Occur in within further Abstraction given in linear input samples complexity.

3. Curvature Tensor:
   This Section Describes a derived mathematical equations From differential equations described as "Curvature tensor", that calculates the edge cases of given variables extracted from a matrix, Its Usage is for example:
   1. Filtering Noise and distingusih complexity of edge case.
   2. Differential sensitive magnitude given from extracted matrix Components such as magnitude sum of x given each vectors are linear or nonlinear complexity inside a matrix of x.
   Code form:
```
trA = k / (1.0 - anisotropy) + eps  
trB = (1/2 + mag_G) / (1.0 + trA**2)
trC = (1/6 + K_G) / (trB**2 - 1.0)
```
Explanation:
1. trA: Given k range is positive and not < 0, the product of trA from division of (1.0 - anisotropy) calculates the complexity of the given k energy with anisotropy of the actual fluctuations of gradient of input x, in which, Anisotropy > 0.75, and k ranging from 2 to 20, or f(k) = 20 < k > 2  indicating a stable moderate complexity of the input and the domain is guaranteed Nonlinear, trA is guaranteed > 0.
2. trB: given the first order sum of (1/2) with mag_G, this part of block is necessary for Normalization of given magnitude of x initialized as mag_G in which mag_G > 0, division of (1.0 + trA**2), indicates that the value of trA**2 increased via sum of +1.0, that has a growth of Non-polynomial meaning it forms a stable sigmoid curve. allowing for better complexity separation mechanism after noise was Filtered.
3. trC: given the second derivative order of (1/6) with K_G, meaning K_G is a "sigmoid" increase of (1.0 + k) projected as such as:
   ```
   K_G = 1.0 / (1.0 + k)
   ```
   allowing for further deriving the sigmoid growth of k given k > 0, division of (1.0 + k) allows for efficient normalization and baseline comparison of how growth of K_G improves over time from a baseline of (1.0 + k).


# Instructions:
To use AWE You must download or import Python library such as:
1. Numpy
2. Sklearn
Note: Supports python 3.14+

^. Step By Step usage:
1. ~ download my AWE Encoder, Plug it in python environment along with Any Numpy MLP SetUp
2. If you're using different Dense layer class, you must define the X_train, This is the X samples of the train_test_split(...), inside to the Dense class initialization.
   Example Initialization of setUp (A must before Trying AWE):
```
   X_train, X_test, y_train, y_test = train_test_split(X, y, ....) # where X is Input.
   Dense = Dense((X_train, input_dim, parameters=5000, ....)) or you can replace X_train with the real Input X.
 ```
  
4. ~ My current MLP Already has the single weight (self.W) plugged with special_weight.weight_encoder(), or you can manually test or add more weights
5. ~ set to learning rate to 0.1 for good balance of learning or lower or higher depends on your needs.
6. ~ Create and import make_classfication() to directly test and dont forget to import train_test_split() too.
7. ~ you're ready to try the weight encoder and see the consistent accuracy. 


# Test Results of my Experiment:
for a convincing results, the data for training i used is train_test_split() in which X, and y, which correlates for input and correct training data set, and random_state is 99.
^. 1. Make_classification samples:
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

   the underlying accuracy consistently ranging from 79% - 82% on 1000 samples with 16 hidden_dim. with Noted Results:
   1. Mean Accuracy Ranging From 92% to 94% consistently accross 10 trials
   2. Std Accuracy Ranging from 8% (0.08) to 14% (0.14) consistently accross 10 trials
   3. Variance Accuracy Ranging from 2% (0.02) to 4% (0.040 consistently accross 10 trials

^. 2. Make_moons samples:
   code form:
 ```
 X, y_raw = make_moons(
    n_samples = 5000, 
    noise=0.5,  
    random_state=99) 
 ```
With 50% noise,
The underlying Accuracy is around 73-76% from epoch 0 to 900 consistently accross 10 different initialization with 5000 samples and 5000 hidden dim parameters. with Noted results:
1. Mean accuracy ranging from 75 - 82% consistently accross 10 trials
2. Std accuracy ranging from 5% (0.05) to 15% consistently accross 10 trials
3. Var accuracy ranging from 0.2% (0.002) to 2% (0.02) consistently accross 10 trials.
   

# Small Limitations Features:

On Make_circles samples initialization, The underlying consistensy of the model was around 54-63% consistently accross 10 trials from epoch 0 to 900, with 5000 parameters and 1000 samples and 50% noise given from this code:
```
X, y_raw = make_circles(
    n_samples = 1000, 
    noise=0.5,  
    random_state=99)
```

with noise 10%, The underlying Accuracy Ranging from 73 - 81% with the same amount of parameters and same 1000 samples
  
# Final Conclusive Results

^. With 5 different trials, and each Mean accuracy on each trial, given this data:
```
baseline_mean_accuracy_each_trial = np.mean([80, 75, 84, 63, 65])
AW_mean_accuracy_each_trial = np.mean([91, 96, 95, 97, 95])

delta = AWE - baseline
print(delta)
```
Note: Each vector there represents the actual mean accuracy percentage of Each trial, Where:
^. 1.  Baseline = Represents The actual mean Accuracy on Regular MLP without AWE.
^. 2. AWE = Represents the actual mean accuracy on Regular MLP with AWE>

Each MLP Was Given with Make_classification results with 1000 samples, the same as above Make_classification, and 5000 Parameters.
The Underlying Mean Delta Accuracy is: 21.4%

