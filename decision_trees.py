import numpy as np
import math
from random import randint
# from test_script import load_data
'''
## Tree representation
    Any node of the tree will be a funcion. Both yes and no branches will be 
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
    tree = {
        n_: (f_0, f_1),
        n_0: (1, 0),
        n_1: (0, f_11),
        n_11: (0, 1)
    }
    Every node will have an associated function, that will be the index of the
    feature that splits that node.
    functions = {
        n_: f1,
        n_0: f3,
        n_1: f4,
        n_11: fn
    }

    A node will be represented as dict, with the question as a function name (str)
    and the decisions as a tuple element. That way, the leaves will be booleans (0 / 1)
'''

class Tree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {'n_': [None, None]} # Initialize tree
        self.node_functions = dict() # Dict to save features that split every node
        self.coords = [] # A list to navigate through the tree when solving all nodes
    

    def _H(self, Y):
        '''Compute the information entropy for a list of labels'''
        _, counts = np.unique(Y, return_counts=True)
        N = sum(counts)
        return sum([(-i/N) * math.log((i/N), 2) for i in counts])


    def _inf_gain(self, X, Y):
        _H = self._H(Y) # Entropy of the whole dataset
        unique_decs, decs_counts = np.unique(X, return_counts=True) # Count decisions
        N = sum(decs_counts)
        decs_H = list() # Decisions entropy list
        # Compute entropy for the different decisions
        for dec in unique_decs:
            y = Y[X == dec] # Labels for feature == decision
            decs_H.append(self._H(y))
        return _H - sum([(i/N) * H for i, H in zip(decs_counts, decs_H)])
        
    
    def _decs_accuracy(self, X, Y):
        '''
        Decision accuracy as follows:
        (Yes-No, Yes-Yes)
        '''
        y_yes = Y[X == 1]
        y_no = Y[X == 0]
        # Accuracy for Yes-Yes, No-No
        yes_yes_acc = (sum(y_yes) + len(y_no) - sum(y_no))/(len(y_no)+len(y_yes))
        # Accuracy for Yes-No, No-Yes
        yes_no_acc = (len(y_yes) - sum(y_yes) + sum(y_no))/(len(y_no) + len(y_yes))
        return (yes_no_acc, yes_yes_acc)
    
    
    def _solve_node(self, XX, Y, coords=[]):
        '''
        This recursion function will iterate over the tree until all nodes are solved, it run out of features
        or reach the max_depth. This function is valid for binary and real data.
        '''
        actual_coords = coords[:] # It seems that coords is mutating trhough the recursive calls, so I have to do this
        for leaf in self.tree['n_'+''.join(map(str, actual_coords))]:
            if leaf in self.node_functions.keys(): # If the solved was already solved 
                continue
            if (leaf is None) or (leaf not in (1, 0)): #  or isinstance(leaf, str)
                if len(np.unique(XX))  > 2: # If the feature is not binary
                    # Calculate IG to determine the best feature and threshold to split, responding to question F < threshold
                    features_range = np.array([np.linspace(min(X), max(X), 12)[1:-1] for X in XX.T]) # Not including max and min
                    IG = []
                    for n in range(10):
                        IG.append([self._inf_gain(X < threshold, Y) for X, threshold in zip(XX.T, features_range.T[n])])
                    IG = np.array(IG)
                    best_feature = np.argmax(IG) % IG.shape[1]
                    best_threshold = features_range[best_feature][np.argmax(IG) // IG.shape[1]]
                    self.node_functions['n_'+''.join(map(str, actual_coords))] = (best_feature, best_threshold) 
                    X = XX.T[best_feature] > best_threshold
                        
                else:
                    IG = [self._inf_gain(X, Y) for X in XX.T]
                    best_feature = IG.index(max(IG)) 
                    self.node_functions['n_'+''.join(map(str, actual_coords))] = best_feature 
                    X = XX.T[best_feature]

                Y01 = (Y[X == 0], Y[X == 1])
                H01 = (self._H(Y01[0]), self._H(Y01[1]))
                
                # If max_depth has been reached, then define by accuracy
                if (len(coords) + 1) == self.max_depth: 
                    decs_acc = self._decs_accuracy(X, Y)
                    self.tree['n_'+''.join(map(str, actual_coords))][1] = decs_acc.index(max(decs_acc))
                    self.tree['n_'+''.join(map(str, actual_coords))][0] = decs_acc.index(min(decs_acc))
                    continue

                # Solve for yes and no
                for i in (0, 1):
                    if H01[i] == 0:
                        self.tree['n_'+''.join(map(str, actual_coords))][i] = Y01[i][0]
                    else:
                        XX_new = XX[X==i]
                        # Check if there will be IG with remainder data
                        IG = [self._inf_gain(X, Y01[i]) for X in XX_new.T]
                        if max(IG) == 0: # Not new information / features
                            # Return the less frequent category of the other branch
                            unique, counts = np.unique(Y01[int(not i)], return_counts=True)
                            self.tree['n_'+''.join(map(str, actual_coords))][i] = int(not unique[list(counts).index(max(counts))])
                            continue
                        self.tree['n_'+''.join(map(str, actual_coords))][i] = 'n_'+''.join(map(str, actual_coords + [i]))
                        self.coords.append(i)
                        self.tree['n_'+''.join(map(str, self.coords))] = [None, None]
                        self._solve_node(XX_new, Y01[i], self.coords)
            else:
                self.coords = []
                break
        

    def train(self, XX, Y):
        if isinstance(XX, list):
            XX = np.array(XX)
        if isinstance(Y, list):
            Y = np.array(Y)
        self._solve_node(XX, Y)
        return(self.tree, self.node_functions)
            


def DT_train_binary(X,Y,max_depth=3):
    '''
    This function simply initializes a DT instance, then it build the tree using
    the train method.
    '''
    DT = Tree(max_depth=max_depth)
    return(DT.train(XX=X, Y=Y))


def DT_make_prediction(x, DT):
    node = 'n_'
    tree_structure = DT[0]
    node_func = DT[1]
    not_binary = isinstance(node_func['n_'], tuple)
    out = None
    while True:
        if out in (1, 0):
            return(out)
        else:
            if not_binary: # If not binary
                feature, threshold = node_func[node]
                out = tree_structure[node][int(x[int(feature)] > threshold)]
                node = out
            else:
                feature = node_func[node]
                out = tree_structure[node][int(x[int(feature)])]
                node = out


def DT_test_binary(X, Y, DT):
    Y_predicted = [DT_make_prediction(x, DT) for x in X]
    hits = [pred == real for pred, real in zip (Y_predicted, Y)]
    return(np.mean(hits))


def DT_train_real(X, Y, max_depth):
    return DT_train_binary(X, Y, max_depth)

def DT_test_real(X, Y, DT):
    return DT_test_binary(X, Y, DT)
