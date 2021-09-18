# Decision Trees
    
    ## DT_train_binary(X,Y,max_depth)
    ...........................................................................
    Any node of the tree will be a string. Both yes and no branches will be 
    represented as tuple element, with index element 0 for NO, and index 
    element 1 for YES. Every node string will start with 'n_' followed by 
    the decision that led to that node. That way, 'n_' will be the initial node,
    'n_0' will be the node for the NO branch of the 'n_' node, 'n_01' will be 
    the node for the YES branch of the 'n_0' node and so on. There will be a node
    function object that will associate every node with the feature index that
    splits that node. The number of characters following 'n_' is equal to the 
    depth of that node.

    So, as an example, consider the following tree:
                 ___node___
                  /      \
                 /        \
                /          \
        ___node___0    ___node___1
          /    \          /    \
         /      \	     /      \
        /        \      /        \
       YES       NO    NO     ___node___11
                                /    \
                               /      \
                              /        \
                             NO        YES
    The tree structure will be represented as follows:
    tree = {
        'n_': (n_0,n_1),
        'n_0': (1, 0),
        'n_1': (0, n_11),
        'n_11': (0, 1)
    }
    In the function object for the nodes, each node will have associated an index to itself. 
    That index is the index of the feature that splits that node. For instance, for the next 
    example, the 'n_1' is split by the feature with index 2, i.e. the third feature.
    functions = {
        'n_': 1,
        'n_0': 3,
        'n_1': 2,
        'n_11': 0
    }

    To implement that, I created a Tree class, with different methods that are called 
    during the training. Those methods are regarding to entropy calculation, information
    gain and accuracy.

    The _solve_node method is the one in charge of solving all the nodes of the decision tree.
    That method follow the next steps to solve the nodes:
        - Take the initialized tree. Any not solved node will be a None. The node is considered
        to be completed solved when its value is 0 or 1.
        - Check if the actual node if already solved. If it is, then move to the next depth.
        - For the actual node, check if the leaf/branch is solved. If it is, then move to the next 
        branch.
        - Calculates IG, best feature to split that node.
        - Check if it's reaching max depth. If it is, then solve by accuracy, if it's not then 
        continue.
        - Check every branch entropy. If branch entropy == 0, then solve it. If it's not then 
        solve that new node by calling itself in a recursive process. After the branch is solved
        it moves to the next, if the former branch is the last, it moves to the next unsolved 
        node. After moving to next step, it checks if there will be any IG, if there won't be
        IG, then it solves the branch based on accuracy taking into account the next or former 
        branch.

    The former process will repeat until all nodes are solved. There is a class attribute (self.coords)
    to keep track of the actual position in the tree, that attribute will reset every time a new unsolved
    nodes search is necesary.

    ## DT_make_prediction(x, DT)
    ...........................................................................
    The prediction is done this way:
        - A sample vector is used as input. 
        - The first node and output is initialized
        - Using the tree structure object, and the node function object, a while loop will be 
        performed to navigate through the tree until a 0, or 1 node is reached.
        - return out

    ## DT_test_binary(X,Y,max_depth)
    ...........................................................................
    It uses two list comprehensions. First one to save the prediction for every sample, and a second
    one to compare the predicted values with the real ones (hits). The out is just the average of the
    hits.
    
    ## DT_train_real(X,Y,max_depth)
    ...........................................................................
    It will have kind of same structure. For every feature the information gain will be tested, using
    different threholds to split the real variable. For that, ten diferent thresholds between the 
    range of the feature range will be tested to find IG. Lets say our feature range goes from 0 to
    10, then the 0, 10 range will be equally split on 10, and those values will be used to create 
    binary labels to evaluate the IG. So, if we have 3 features, instead of having a [3,] IG vector
    we will have a [3, 10] IG matrix. The best split will be ith feature on the jth threshold so that 
    the ith, jth element of the IG matrix is the maximum. On the function_node object, insted of having
    the feature index, we will have a (feature index, feature threshold) tuple.

    The _solve_node method identifies if the feature is binary itself. If the feature is not binary,
    then it will estimate the best feature, threshold pair to create the question that will generate
    binary features.

    Then, the DT_train_binary will work even if the data is real, since the _solve_node method of the
    Tree class will notice that the features are not binary.

    ## DT_test_real(X,Y,DT)
    ...........................................................................
    This function just call the DT_test_binary function. The DT_make_prediction function knows if
    the features are binary or not, then it doesn't matter if it's not a binary feature, it'll make the
    prediction, and the DT_test_binary function will work too.

# Nearest Neighbors

    ## KNN_test(X_train,Y_train,X_test,Y_test,K)
    ...........................................................................
    I simply used some numpy and scipy.spatial functions so follow the steps of a KNN:
        - Distance matrix is calculated, that is the distances from every X_test to every X_train. 
        - Get the index of KNN
        - Get the KNN values 
        - Get the predicted value by getting the sign of the addition of the KNN values for every sample
        - Build a hits vector, the mean of this vector is the accuracy

    ## choose_K(X_train,Y_train,X_val,Y_val)
    ...........................................................................
    The function performs the following steps:
        - Find posible Ks, that's the [1, N_samples) so that K % 2 == 1.
        - Evaluate the accuracy for every K
        - Find the K with the higher accuracy.
        - If there is more than one best K, it will select the greater one.

# K-Means Clustering

    ## K_Means(X,K,mu)
    ...........................................................................
    The function performs the next steps:
    - Verify K is greater than 0, and that both X and mu are numpy arrays
    - If mu is empty, then it intializes mu as a K size slice of randomly shuffled X. 
    - Initializes the mu_prev variable to keep track of the previous mu
    - Do the next while mu_prev is different from mu
            + Initializes the clusters as a K, Nsamples, Nfeatures shape empty array
            + Calculate the distance matrix between the samples and the mus
            + Assign every sample to the closest cluster/mu
            + Redefine the previous mu as the actual one
            + Redefine actual mu as the mean of every cluster using nanmean
    - Return mu vector