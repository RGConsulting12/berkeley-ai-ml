Gradient descent is an optimization algorithm which is commonly-used to train machine learning models and neural networks. It trains machine learning models by minimizing errors between predicted and actual results.
Training data helps these models learn over time, and the cost function within gradient descent specifically acts as a barometer, gauging its accuracy with each iteration of parameter updates. Until the function is close to or equal to zero, the model will continue to adjust its parameters to yield the smallest possible error. Once machine learning models are optimized for accuracy, they can be powerful tools for artificial intelligence (AI) and computer science applications.

How does gradient descent work?

Before we dive into gradient descent, it may help to review some concepts from linear regression. You may recall the following formula for the slope of a line, which is y = mx + b, where m represents the slope and b is the intercept on the y-axis.

You may also recall plotting a scatterplot in statistics and finding the line of best fit, which required calculating the error between the actual output and the predicted output (y-hat) using the mean squared error formula. The gradient descent algorithm behaves similarly, but it is based on a convex function.

The starting point is just an arbitrary point for us to evaluate the performance. From that starting point, we will find the derivative (or slope), and from there, we can use a tangent line to observe the steepness of the slope. The slope will inform the updates to the model parameters—i.e. the weights and bias. The slope at the starting point will be steeper, but as new parameters are generated, the steepness should gradually reduce until it reaches the lowest point on the curve, known as the point of convergence.

Similar to finding the line of best fit in linear regression, the goal of gradient descent is to minimize the cost function, or the error between predicted and actual y. In order to do this, it requires two data points—a direction and a learning rate. These factors determine the partial derivative calculations of future iterations, allowing it to gradually arrive at the local or global minimum (i.e. point of convergence).

Learning rate (also referred to as step size or the alpha) is the size of the steps that are taken to reach the minimum. This is typically a small value, and it is evaluated and updated based on the behavior of the cost function. High learning rates result in larger steps but risks overshooting the minimum. Conversely, a low learning rate has small step sizes. While it has the advantage of more precision, the number of iterations compromises overall efficiency as this takes more time and computations to reach the minimum.
The cost (or loss) function measures the difference, or error, between actual y and predicted y at its current position. This improves the machine learning model’s efficacy by providing feedback to the model so that it can adjust the parameters to minimize the error and find the local or global minimum. It continuously iterates, moving along the direction of steepest descent (or the negative gradient) until the cost function is close to or at zero. At this point, the model will stop learning. Additionally, while the terms, cost function and loss function, are considered synonymous, there is a slight difference between them. It’s worth noting that a loss function refers to the error of one training example, while a cost function calculates the average error across an entire training set.

Types of gradient descent

There are three types of gradient descent learning algorithms: batch gradient descent, stochastic gradient descent and mini-batch gradient descent.
Batch gradient descent

Batch gradient descent sums the error for each point in a training set, updating the model only after all training examples have been evaluated. This process referred to as a training epoch.

While this batching provides computation efficiency, it can still have a long processing time for large training datasets as it still needs to store all of the data into memory. Batch gradient descent also usually produces a stable error gradient and convergence, but sometimes that convergence point isn’t the most ideal, finding the local minimum versus the global one.
Stochastic gradient descent

Stochastic gradient descent (SGD) runs a training epoch for each example within the dataset and it updates each training example's parameters one at a time. Since you only need to hold one training example, they are easier to store in memory. While these frequent updates can offer more detail and speed, it can result in losses in computational efficiency when compared to batch gradient descent. Its frequent updates can result in noisy gradients, but this can also be helpful in escaping the local minimum and finding the global one.
Mini-batch gradient descent

Mini-batch gradient descent combines concepts from both batch gradient descent and stochastic gradient descent. It splits the training dataset into small batch sizes and performs updates on each of those batches. This approach strikes a balance between the computational efficiency of batch gradient descent and the speed of stochastic gradient descent.

Challenges with gradient descent

While gradient descent is the most common approach for optimization problems, it does come with its own set of challenges. Some of them include:
Local minima and saddle points

For convex problems, gradient descent can find the global minimum with ease, but as nonconvex problems emerge, gradient descent can struggle to find the global minimum, where the model achieves the best results.

Recall that when the slope of the cost function is at or close to zero, the model stops learning. A few scenarios beyond the global minimum can also yield this slope, which are local minima and saddle points. Local minima mimic the shape of a global minimum, where the slope of the cost function increases on either side of the current point. However, with saddle points, the negative gradient only exists on one side of the point, reaching a local maximum on one side and a local minimum on the other. Its name inspired by that of a horse’s saddle.

Noisy gradients can help the gradient escape local minimums and saddle points.
Vanishing and Exploding Gradients

In deeper neural networks, particular recurrent neural networks, we can also encounter two other problems when the model is trained with gradient descent and backpropagation.

Vanishing gradients: This occurs when the gradient is too small. As we move backwards during backpropagation, the gradient continues to become smaller, causing the earlier layers in the network to learn more slowly than later layers. When this happens, the weight parameters update until they become insignificant—i.e. 0—resulting in an algorithm that is no longer learning.

Exploding gradients: This happens when the gradient is too large, creating an unstable model. In this case, the model weights will grow too large, and they will eventually be represented as NaN. One solution to this issue is to leverage a dimensionality reduction technique, which can help to minimize complexity within the model.

What is stochastic gradient descent?

Stochastic gradient descent (SGD) is an optimization algorithm commonly used to improve the performance of machine learning models. It is a variant of the traditional gradient descent algorithm, with a key modification: instead of relying on the entire dataset to compute the gradient at each step, SGD uses a single data sample at a time.
Gradient descent

Gradient descent (GD) is an optimization algorithm that iteratively minimizes an objective function. In the context of machine learning (ML), gradient descent is fundamental to improving the performance of supervised learning models during their training phase. Machine learning models, like neural networks, are complex, nonlinear and high-dimensional. Hence, there is no normal equation for such models that can compute the optimal weights, unlike in linear regression. Instead, approximation methods like the variants of gradient descent, Newton’s methods and expectation maximization can be used, among others.

Every model has a loss function, sometimes called a cost function. This function measures how far a model’s predictions are from the true data points. Think of this as a measure of how “wrong” the model’s predictions are. For example, the mean-squared error often serves as the loss function in regression problems. The model training phase is designed to find the parameter values that minimize this loss. Gradient descent is often the optimization technique used in training for this reason. The algorithm computes the gradient, or the slope, of the loss with respect to the model’s parameters. With this gradient, it then takes a step in the opposite direction to reduce the loss. The learning rate (also referred to as step size or the alpha) is the size of the steps and it remains fixed for all model parameters. This process repeats until the model achieves convergence near a minimum.

Convergence ideally occurs at the global minimum. In the following visualization, you can see that the loss value is lower at a local minimum than in its immediate surrounding area, but not necessarily the lowest value overall. The global minimum is the absolute lowest value of the loss function across its entire domain, representing the best possible solution for the problem.

If the learning rate is not small enough, the algorithm will often converge at a local minimum. A well-chosen rate is essential for minimizing the loss function and achieving convergence at a global minimum.

From GD to SGD

The key differentiator between traditional gradient descent and stochastic gradient descent is that SGD updates model weights by using a single training example at a time. The example is randomly picked at each iteration.1 Gradient descent uses the entire training dataset to compute the gradient before each parameter update. This difference in data usage is what makes SGD much less computationally expensive and easier to scale for large datasets. Alternatively, the convergence behavior of SGD is noisier than the noise of GD because the one example datapoint might not be a good representation of the dataset. This misrepresentation updates the points in a slightly “wrong” direction. However, this randomness is what makes SGD faster and sometimes better for nonconvex optimization problems because it can escape shallow local minima, or saddle points.

Strictly speaking, SGD was originally defined to update parameters by using exactly one training sample at a time. In modern usage, the term “SGD” is used loosely to mean “minibatch gradient descent,” a variant of GD in which small batches of training data are used at a time. The major advantage to using subsets of data rather than a singular sample is a lower noise level, because the gradient is equal to the average of losses from the minibatch. For this reason, minibatch gradient descent is the default in deep learning. Contrarily, strict SGD is rarely used in practice. These terms are even conflated by most machine learning libraries such as PyTorch and TensorFlow; optimizers are often called “SGD,” even though they typically use minibatches.

The following illustration provides a clearer depiction of how increasing the sample size of training data reduces oscillations and “noise.”

There are several other variants of GD that are built on basic gradient descent by adding mechanisms to improve speed, stability and convergence.
Momentum-based methods:

By accumulating momentum in dimensions with consistent gradients and dampening updates in dimensions with changing gradients, momentum helps SGD converge faster and with less oscillation.2

Momentum gradient descent: Incorporates a “velocity” term, an average of previous gradients that gives more importance to recent ones. This approach reduces zigzagging, or oscillations, helping the algorithm move faster in the right direction.

NAG (Nesterov accelerated gradient): An improved momentum method that speeds up and smooths convergence by “looking ahead” at where the parameters are headed before computing the gradient. In other words, it anticipates the future gradient and uses this information to inform the current update step.3
Adaptive learning rate methods:

Adaptive learning rate methods, such as AdaGrad and RMSProp, are unique in that they adapt the learning rate for each parameter individually. This approach is in contrast to SGD methods, which use a fixed learning rate for all parameters.

AdaGrad (adaptive gradient algorithm): Adapts the learning rate for each parameter based on its previous gradients. Features that appear less often receive higher learning rates, and frequent features receive lower rates. This approach means that infrequent features are learned quicker than with SGD. This adaptive learning rate means it is a great method for natural language processing (NLP) and recommendation systems with sparse data, in which there is a large discrepancy in feature frequency.2
 
RMSProp (Root Mean Square Propagation): Another adaptive learning rate optimization technique that scales the learning rate for each parameter by using a moving average of recent squared gradients. Past gradient knowledge is discarded and only current gradient knowledge is preserved.4 The learning rate becomes larger for parameters with small gradients and smaller for those with large gradients. This method eliminates the diminishing learning rate problem with AdaGrad. RMSProp helps keep training stable in deep learning, especially for models like recurrent neural networks (RNNs), and it works well on problems where the objective keeps changing, such as in reinforcement learning.
Hybrid methods:

Adam (adaptive moment estimation): Combines momentum-based GD with RMSProp by tracking both the past gradients and the average of squared gradients.4 This combination allows for a fast convergence rate even for noisy and sparse datasets.3 Additionally, the default hyperparameters like a learning rate of 0.001 in many frameworks, work well right away. For very large-scale datasets, however, SGD with momentum can lead to better generalization. The aggressive per-parameter adjustments of Adam can result in overfitting of the training data or settling into sharp minima that don’t generalize as well.
SGD and other GD variants are useful when training time is the bottleneck.5

| Variant         | Data used per step        | Key feature                          | Common use        |
|-----------------|---------------------------|--------------------------------------|-------------------|
| GD              | All data                  | Stable but slow                      | Small datasets    |
| SGD             | 1 sample (classic SGD)    | Noisy but fast                       | Online learning   |
| Mini-Batch GD   | Few samples               | Balanced and scalable                | Deep learning     |
| Momentum        | Batch / mini-batch        | Accelerates in right direction       | Neural nets       |
| NAG             | Batch / mini-batch        | Look-ahead momentum                  | Faster convergence|
| AdaGrad         | Mini-batch                | Adaptive learning rates              | Sparse data       |
| RMSProp         | Mini-batch                | Fixes AdaGrad decay                  | RNNs, deep nets   |
| Adam            | Mini-batch                | Momentum + RMSProp                   | Default choice today |



Simple Python implementation of SGD

When working with machine learning frameworks, there are built-in SGD optimizer classes one can use. For example,   torch.optim.SGD   for PyTorch,   tf.keras.optimizers.SGD   for Keras built into TensorFlow, and   SGDRegressor   for Scikit-learn.

For learning purposes, let’s walk through a simple Python implementation of SGD from scratch.

To reiterate, our objective is to find the best parameters (model weights) that minimize the loss function (a measure of how wrong our predictions are). We will update one sample at a time or a very small batch size.

To start, we can initialize the parameter values (weights) randomly. Next, we can select a random data point 
(x,y). From there, we compute the prediction and the error. For this simple demonstration, let’s try to fit a simple line: y=mx+b.

The next step in the process is backpropagation, in which the gradients of the loss function are computed with respect to the parameters. These gradients (derivatives) are then used to update the parameters during the SGD optimization process. Because the gradient points to the direction of increase of the loss function, SGD subtracts each gradient from its respective current parameter value. We can think of this as moving in the opposite direction of the gradient to decrease the loss function. Hence, the “descent” in stochastic gradient descent. We repeat these steps until a fixed number of epochs or once the loss is less than the tolerance. The latter would mean that the loss is hardly changing and no longer are we improving the objective function. In other words, we stop once the algorithm converges.

import numpy as np 

 

def stochastic_gradient_descent(X, y, lr=0.01, epochs=100, tol=1e-6): 

    “”” 

    Perform Stochastic Gradient Descent (SGD) to fit a line y = w*x + b 

     

    Parameters: 

        X (ndarray): Input features 

        y (ndarray): Target values 

        lr (float): Learning rate (step size for updates) 

        epochs (int): Number of iterations through the dataset 

     

    Returns: 

        w (float): Learned weight 

        b (float): Learned bias 

    “”” 

    # Initialize parameters randomly 

    w = np.random.randn() 

    b = np.random.randn() 

     

    n = len(X) 

 

    prev_loss = float(‘inf’) 

     

    for epoch in range(epochs): 

        # Shuffle the data for each epoch 

        indices = np.arange(n) 

        np.random.shuffle(indices) 

         

        for i in indices: 

            xi = X[i] 

            yi = y[i] 

             

            # Prediction 

            y_pred = w * xi + b 

             

            # Compute gradients (derivatives) 

            dw = -2 * xi * (yi - y_pred)   # derivative wrt w 

            db = -2 * (yi - y_pred)        # derivative wrt b 

             

            # Update parameters 

            w -= lr * dw 

            b -= lr * db 

         

        

        # Compute loss at the end of the epoch 

        loss = np.mean((y - (w*X + b))**2) 

         

        # Check stopping condition 

        if abs(prev_loss - loss) < tol: 

            print(f”Stopped early at epoch {epoch+1}”) 

            break 

                 

        prev_loss = loss 

             

    return w, b



Applications of SGD

SGD is the most common optimization method for training deep neural networks. In deep learning, a subset of machine learning within the broader field of data science, the objective is for computers to simulate the complex decision-making power of the human brain. Traditional ML models use simple neural networks consisting of one or two layers. Whereas deep learning models use three or more layers. Typically, hundreds or thousands of layers are needed to train the models. Given SGD’s ease to scale for large training sets, it is often the go-to approach for training deep neural networks. Other applications of SGD training include ridge regression, regularized logistic regression and the optimization of the hinge loss function used in support vector machines (SVMs) with a linear kernel.
Conclusion

SGD is a variant of GD that minimizes a machine learning model’s loss function by using a single data sample at a time. This approach is unlike GD, which depends on the entire dataset at each step to compute the gradient. There are several other GD variants that can be grouped as momentum-based or adaptive learning methods. Momentum gradient descent and Nesterov accelerated gradient are examples of the former. These methods leverage accumulated momentum in dimensions with consistent gradients and dampen updates in dimensions with changing gradients. Thus, helping SGD converge faster and with less oscillation. Adaptive learning rate methods such as AdaGrad and RMSProp adapt the learning rate for each parameter individually, unlike traditional SGD, which uses a fixed learning rate. In addition, hybrid methods like Adam offer a powerful alternative by combining the strengths of momentum-based GD and RMSProp.