# Spectral-Grounding: Specialized-MLP-for-extreme-noise-robustness
A Custom specialized MLP Designed to handle noise with a very consistent Accuracy on 1000+ Samples. up to 95% consistent Accuracy on 10 trials (10 different initialization), each trial consist of 900 epoch trainings. Using a Highly Specialized Custom Module Called "Abstract Weight Encoder", or short as AWE.
Note: The MLP tested here doesn't use any kind of Optimizer for Training and Generalization.

# MLP Introduction

Multilayer Perceptron (MLP) is a foundational, supervised feed-forward artificial neural network consisting of at least three layers (input, hidden, output) of fully connected neurons. It uses nonlinear activation functions (like ReLU or sigmoid) and backpropagation to learn complex, non-linearly separable relationships, commonly used for classification and regression tasks. 

^. MLP and Setup Requirements:
1. ~ Numpy and sklearn libraries
2. ~ 16-5000 hidden dimension (parameters)
3. ~ Input dim and output dim depends on sklearn samples (Mostly 1000-10000 samples)

# How AWE Works:
AWE is a specialized custom weight shaping method that used eigenvalue and spectral methods to calculate covariance inside a given input data, and shape the correct Weight from the given eigenvalue, AWE Works by processing input and then captures the necessary eigenvalue to shape a properly initialized Weight that aligns with input data complexity, So, MLP training will be much more consistent and robust against noise. The reason why eigenvalue works robustly against noise and causes great consistency, Happens for a few Reasons:

● 1. Eigenvalue captures the necessary continual pattern inside a given input data:
   • By using Eigenvalue, the necessary pattern inside a given input data could be captured by using Eigenvalue equations .

2. Eigenvalue captures the necessary standard Complexity of the input variance:
   • Eigenvalue also captures the input complexity inside a given matrix, allowing it to properly captures the standard variance necessary to be used for properly initialized weight.

3. Eigenvalue provides discrete covariance of a given input data, allowing it to shapes weight:
  • Eigenvalue also provides deeper discrete covariance results that can determine when an input has strong geometric complexity enough that the covariance is properly consistent necessary for weight shaping, this allows the model to be much more flexible, and does'nt get constrained by 10x more samples than parameters rules.


^. Experiment Note:
The MLP i used is mostly small basic Numpy MLP, and its independent, meaning it doesn't use Dropout or any helper modules from pytorch. To further test the capabilites of the abstract weight encoder (AWE) in smaller Datasets and fewer Parameters, Parameters can be scaled, and it doesn't cause Accuracy or generalization degradation, with a very consistent results accross different generalization samples, especially On linear Make_classification module with accuracy being as consistent as much as 90-95% Accuracy on training with features:
1. class separation = 1.5
2. random_state = 99

And around 91-93% generalization accuracies consistently on sklearn make_classification() syntethic datas with minimal neccessary noise to capture average real world noise complexity consistently, using only 100 parameters and 1000 samples for fair comparisons.



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
   AME is a fundamental equations needed for calculating further Abstraction error given the magnitude and the gradient of the input, derived and inspired from KL divergence. further modelling a neccessary error given the complexity of input samples. its a necessary component given to calculate efficient distributed complexity.

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
Explanation: the log(x_mag + 1) Provides a log value by a non-polynomial function of X_mag in order to express a finite sum of terms consisting of constants and variables raised to whole Number exponents to help identify the logarithmic scale of X_mag. and multiplcation with log(1 + grad_energy) to normalize the range of AME to > 0, given the positive value of each variables.

in Which High AME, AME > 0.75. Correlates towards such High Error indication of Possible Ongoing Abstraction due to the Complexity of The input samples. In Which Low AME Correlates towards More Efficient Abstraction and Low Possible Error can Occur in within further Abstraction given in linear input samples complexity.

3. Curvature Tensor:
   This Section Describes a derived and inspired mathematical equations From differential equations and taylor expansion series, described as "Curvature tensor", that calculates the edge cases of given variables extracted from a matrix, Its Usage is for example:
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
2. trB: given the first order sum of (1/2) with mag_G,Where mag_G is:
```
mag = np.mean(np.linalg.norm(x, axis=-1))
mag_G = 1.0 / (1.0 + mag)
``` 
this part of block is necessary for Normalization of given magnitude of x initialized as mag_G in which mag_G > 0, division of (1.0 + trA**2), indicates that the value of trA**2 increased via sum of +1.0, that has a growth of Non-polynomial meaning it forms a stable sigmoid curve. allowing for better complexity separation mechanism after noise was Filtered.

3. trC: given the second derivative order of (1/6) with K_G, meaning K_G is a "sigmoid" increase of (1.0 + k) projected as such as:
   ```
   K_G = 1.0 / (1.0 + k)
   ```
   allowing for further deriving the sigmoid growth of k given k > 0, division of (1.0 + k) allows for efficient normalization and baseline comparison of how growth of K_G improves over time from a baseline of (1.0 + k).

4. AEL (abstraction efficiency limit):

AEL is an equations used to calculate a possible limitation on how the model could do such neccessary abstraction given the input complexity. Derived and inspired from differential geometry, given AEL code form:
```
AEL = 0.3 + spectral_similarity * anisotropy
```
Note:
In which anisotropy in data and information refers to the property where data characteristics, spatial dependencies, or physical properties vary depending on the direction or angle of measurement. And spectral similarity is the similarity of which a weight and other possible similar weights that almost has the same spectral complexity, given both range (0 -> 1.0)

The empirical baseline of 0.3 refers to the barrier between non linear anisotropy which range is > 0.3, and linear anisotropy which is < 0.3, used as a neccessary directional anchor neccessary to include possible regime change in anisotropy in directional covariance.

5. EDC (Efficient distributed complexity):
EDC is another equations derived from differential equations to calculate the final complexity in which a weight will be encoded, given code form:
```
AME_sigmoid_growth = 1.0 / (1.0 + np.exp(-AME))
EDC = k + AEL * (1.0 - AME_sigmoid_growth)
```

Note:
This equations capture the relevant component neccessary for capturing input complexity from 3 angles, its compositional dimensional complexity (k), abstraction limitation (AEL) and decrease of possible AME (1.0 - AME).

K is neccessary to capture relevent information to how an input complexity using binary search that results in int (k) range > 0. And AEL is neccessary to capture the limitation based on its environment of the input itself, and (1.0 - AME) to calculate a decrease of AME given the input complexity in a changing environment. 
Together they act as a neccessary encoder in which the range of EDC is > 0, given both variables are positive, and also provided a log2 value of the directional covariance of the input (log²(DirCov)) in which neccessary to break the symmetric relationships between the weights and input for neccessary abstraction. 

# Usage instructions:
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
  
4. ~ My current MLP Already has the single weight (self.W) plugged with special_weight.weight_encoder(), or you can manually test and add more weights
   Ex setUp:
```
   self.W = self.special_weight(input_size, output_size).weight_encoder(X) # where X is input.
   self.W1 = .... # same code as self.W
```
 
6. ~ set to learning rate to 0.1 for good balance of learning or lower or higher depends on your needs.
7. ~ Create and import sklearn make_classfication() to directly test and dont forget to import train_test_split() for training too.

9. For further test of generalizations after trainings, you can create a more robust realistic data sets, or you can directly copy my realistic_data_sets() function and add_distribution_shift() function in my MLP-SetUp code.

8. ~ you're ready to try the weight encoder and see the consistent accuracy. 


# Test Results of my Experiment:
for a convincing results, the data for training i used is train_test_split() in which X, and y, which correlates for input and correct training data set, and random_state is 99 and 123 for testing.
This block only includes Training results.

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
   the underlying Training accuracies reaches up to 86%-93% on 1000 samples consistent accross  10 trials (10*1000 epochs) with 16 hidden_dim, in which each trial resets the models training to first initialization. and 93-97% accuracy on up to 2000 hidden_dim with 1000 samples consistently accross 10 trials.
   
   ~ When The model was trained using X_noisy, in which X_noisy is a distorted input in which X + noise_scale, where noise_scale is 0.9 (90% noise) using np.random.uniform(0, noise_scale, size=x.shape), and it was put inside 
```
train_test_split(X_noisy, y, test_size=0.9, random_state=99)
```

   the underlying Training accuracies consistently ranging from 79% - 82% on 1000 samples with 16 hidden_dim. with Noted Results:
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
The underlying Training Accuracies is around 73-76% from epoch 0 to 900 consistently accross 10 different initialization with 5000 samples and 5000 hidden dim parameters. with Noted results:
1. Mean accuracy ranging from 75 - 82% consistently accross 10 trials
2. Std accuracy ranging from 5% (0.05) to 15% consistently accross 10 trials
3. Var accuracy ranging from 0.2% (0.002) to 2% (0.02) consistently accross 10 trials.
   


# Final Conclusive Results
1. For trainings results:

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
The Underlying Mean Delta Accuracy is:

• 21.4%

2. For generalization capabilities:

The underlying test accuracies (after trainings) of AWE MLP using 100 parameters and 1000 make_classifications samples, Are very consistent given in this average results after 10 trials:

1. Standard split:
- Mean = 0.916
- Std = 0.018
- Min = 0.886
- Max = 0.943

Note: Standard split is regular Accuracy score given the model predictions based on its capabilities of distinguishing basic noise and real features.

2. Noise injection:
- Mean = 0.912
- Std = 0.20
- Min = 0.878
- Max = 0.942

Note: Noise injection is a test accuracy in which the X input was corrupted with slightly controlled unncessary noise to see how well the model can predict what's unnecessary noise and whats real features

3. distribution shift:
- Mean = 0.831
- Std = 0.022
- Min = 0.800
- Max = 0.856

Note: Distribution shift is a neccessary and important test to see How well the model recognize Real world complex noise in which neccessary to see wether a Given model overfits or not.

●. Results Conclusion:
This Further Proves that AWE MLP, without any helper module like Dropout, etc. Can still perform very Well on noisy, complex environment consistently accross 10 trials.

 

3. For both trainings and test validations:

The underlying mean Accuracy after Training and tested on synthetic data such as make_classification samples, total 1000 samples with 100 parameters, Is:

• 87-91%. 

Given code form:
```
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.9, random_state=99)
_, X_test2, _, y_test2 = train_test_split(X, y, test_size=0.9, random_state=123)

test_accuracy = model.predict(X_test, y_test, epochs=1000)
test_accuracy = model.predict(X_test2, y_test2, epochs=1000)

```

Meaning that both testing after trainings, tested for 2000 epochs, the STD variance is only 0.7% variance of accuracy. meaning the underlying accuracy of each 1000 epochs of testing varies only a slight 0.7 -> 2.1%, because each 1000 epochs was tested on different samples.


# Small Limitations Features:

On Make_circles samples initialization, The underlying consistensy of the model was around 54-63% consistently accross 10 trials from epoch 0 to 900, with 5000 parameters and 1000 samples and 50% noise given from this code:
```
X, y_raw = make_circles(
    n_samples = 1000, 
    noise=0.5,  
    random_state=99)
```

with noise 10%, The underlying Training Accuracies Ranging from 73 - 81% with the same amount of parameters and same 1000 samples
 
●. From all of this results, the main conclusion resides in how AWE MLP is used and behave in noise complex environment, making it able to distinguish features from noise a lot better than baseline MLP with higher consistency results.

●. AWE also doesn't necessarily prevent overfitting and underfitting, it helps to reduce it, by making it more stable during training and more consistent at learning during generalization.



