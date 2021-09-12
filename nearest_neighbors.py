import numpy as np 
from scipy import spatial

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    '''
    Gets a X and Y train set to evaluate the accuracy of KNN prediction over X and Y test set
    '''
    distance_matrix = spatial.distance_matrix(X_train, X_test)
    KNN_indexes = np.argsort(distance_matrix)[:,:K]
    KNN_values = np.array([Y_train[indexes] for indexes in KNN_indexes])
    Y_predicted = np.apply_along_axis(sum, axis=1, arr=KNN_values)
    hits = np.array([Y_real == Y_pred for Y_real, Y_pred in zip(Y_test, Y_predicted)])
    return np.mean(hits)

def choose_K(X_train, Y_train, X_val, Y_val):
    '''
    Assess the best K as the most accurate on the validation set
    '''
    posssible_Ks = [K for K in range(1, X_train.shape[0]) if K % 2 == 1]
    Ks_accuracy = np.array([KNN_test(X_train, Y_train, X_val, Y_val, K) for K in posssible_Ks])
    # I inverted that so as that if K=1 is the best K, and there is another K with same accuracy it selects the K != 1
    return posssible_Ks[::-1][Ks_accuracy[::-1].argmax()]