# Decision Trees
    
    ## DT_train_binary(X,Y,max_depth)
    ...........................................................................
    Any node of the tree will be a function. Both yes and no branches will be 
    represented as tuple element, with index element 0 for no, and index 
    element 1 for yes. Every node is a key of the dict, and every branch
    could be a final answer o the name of the next node.

    So, as an example, consider the next tree:
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
    The tree structure will be represented as follow:
    tree = {
        n_: (f_0, f_1),
        n_0: (1, 0),
        n_1: (0, f_11),
        n_11: (0, 1)
    }
    Every node will have an associated function, that will be the index of the
    feature that splits that node in the case of binary features.
    functions = {
        n_: f1,
        n_0: f3,
        n_1: f4,
        n_11: fn
    }

    A node will be represented as dict, with the question/split feature as a function name (str)
    and the decisions as a tuple element. That way, the leaves will be booleans (0 / 1)
    So, the tree object will be a tuple containing the tree structure and the node function,
    which is the feature index for binary labels.

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
        IG, then solve the branch in function of next or former branch out or most accurate out.

    The former process will repeat until all nodes are solved. There is a class attribute (self.coords)
    to keep track of the actual position in the tree, that attribute will reset every time a new unsolved
    nodes search is necesary.
    
    The function to implement uses the train method over the Tree instance that was previously
    defined.

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
        - Get the predicted value by adding the KNN values for every sample
        - Build a hits vector, the mean of this vector is the accuracy

    ## choose_K(X_train,Y_train,X_val,Y_val)
    ...........................................................................
    The function performs the following steps:
        - Find posible Ks, that's the [1, N_samples) so that K % 2 == 1.
        - Evaluate the accuracy for every K
        - Find the K with the higher accuracy.
        - If there is more than one best K, it will select the greater one.