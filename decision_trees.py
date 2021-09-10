# %%
import numpy as np
import math
from random import randint

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

class Tree(dict):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {'n_': [None, None]}
        self.solved_tree = {'n_': [None, None]}
        self.node_functions = dict()
        self.coords = []

    def _node(self, decision_vector=[1]):
        '''
        It retrieves the node / leaf given a decision vector. For the example
        tree, the decision vector (0, 0) would return YES, and (1, 1, 0) would return NO.
            decision_vector: tuple or list
        '''

        node_function = 'n_' # String that identifies the function for that node
        tmp_obj = self.tree[node_function] # Temporal object to save the tree below that node
        
        for depth, coord in enumerate(decision_vector):
            if isinstance(tmp_obj, dict):  # If there's a question to make
                tmp_obj = tmp_obj[node_function] # Drop the question and get choises
                tmp_obj = tmp_obj[coord] # Make a choise
            elif isinstance(tmp_obj, tuple): # If there is no question to evaluate
                tmp_obj = tmp_obj[coord] # Make a choise
            else:
                break # Final choise is ready
            node_function += str(coord) # Next node function
        return tmp_obj
    
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
        IG = [self._inf_gain(X  , Y) for X in XX.T]
        best_feature = IG.index(max(IG))
        self.node_functions['n_'+''.join(map(str, coords))] = best_feature
        X = XX.T[best_feature]
        Y0 = Y[X == 0]
        Y1 = Y[X == 1]
        H0 = self._H(Y0)
        H1 = self._H(Y1)
        if H0 == 0:
            self.tree['n_'+''.join(map(str, coords))][0] = Y0[0]
        else:
            XX_new = XX[X==0]
            self.tree['n_'+''.join(map(str, coords))][0] = 'n_'+''.join(map(str, coords + [0]))
            self.node_functions[self.tree['n_'+''.join(map(str, coords))][0]] = best_feature
            self.coords.append(0)
            self.tree['n_'+''.join(map(str, self.coords))] = [None, None]
            self._solve_node(XX_new, Y0, self.coords)
        if H1 == 0:
            self.tree['n_'+''.join(map(str, coords))][1] = Y1[0]
        else:
            XX_new = XX[X==1]
            self.tree['n_'+''.join(map(str, coords))][0] = 'n_'+''.join(map(str, coords + [1]))
            self.node_functions[self.tree['n_'+''.join(map(str, coords))][0]] = best_feature
            coords.append(1)
            self.tree['n_'+''.join(map(str, self.coords))] = [None, None]
            self._solve_node(XX_new, Y1, self.coords)
        print(self.tree, self.node_functions)



    def train(self, XX, Y):
        f_key = 'n_'
        depth = 0
        IG = [self._inf_gain(X, Y) for X in XX.T]
        best_feature = IG.index(max(IG))

        self.node_functions
        return (IG, best_feature)
            


def DT_train_binary(XX,Y,max_depth=3):
    '''
    X: 2D ndarray, binary features
    Y: 1D ndarray, binary target
    max_depth: int, maximum depth of the tree. If max_depth=-1, there's no max depth
    '''

# %%
DT = Tree()
# DT.tree = tree
Y = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0])
XX = np.array([
    [1, 1, 0, 0], 
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 1 ,1],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 1, 1]
])
# %%
DT._solve_node(XX, Y)
# %%
