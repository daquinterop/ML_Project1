# %%
import numpy as np
import math
from random import randint

'''
## Tree representation
    Any node of the tree will be a funcion. Both yes and no branches will be 
    represented as tuple element, with index element 0 for no, and index 
    element 1 for yes. So, any depth of the tree will be a dict, with 
    functions as keys and tuples as items.

    So, as an example, consider the next tree:
                function
                 /      \
                /        \
               /          \
            fuction_0    function_1
              /    \          /    \
             /      \	     /      \
            /        \      /        \
            YES       NO    NO     function_11
                                     /    \
                                    /      \
                                   /        \
                                  NO        YES
    tree = {    
        'f_0': ({
            'f_00': (         # No depth 1
                1,                    # No depth 2
                0                     # Yes depth 2    
            )}, # Depth 2
            {'f_01': (        # Yes depth 1
                0,                    # No depth 2
                {'f_011': (    # Yes depth 2
                    0,                # No depth 3
                    1                 # Yes depth 3
                    )}
                ) # Depth 3 
            } # Depth 2
        ) # Depth 1
    } 

    A node will be represented as dict, with the question as a function name (str)
    and the decisions as a tuple element. That way, the leaves will be booleans (0 / 1)
'''

class Tree(dict):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = dict()
        self.node_functions = dict()

    def _node(self, decision_vector=[1]):
        '''
        It retrieves the node / leaf given a decision vector. For the example
        tree, the decision vector (0, 0) would return YES, and (1, 1, 0) would return NO.
            decision_vector: tuple or list
        '''

        node_function = 'f_' # String that identifies the function for that node
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
    
    def _solve_node(self, XX, Y):
        IG = [self._inf_gain(X, Y) for X in XX.T]
        best_feature = IG.index(max(IG))
        X = XX.T[best_feature]
        decs_acc = self._decs_accuracy(X, Y)
        yes_desc = decs_acc.index(max(decs_acc))



    def train(self, XX, Y):
        f_key = 'f_'
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
tree = {    
    'f_': ({
        'f_0': (         # No depth 1
            1,                    # No depth 2
            0                     # Yes depth 2    
        )}, # Depth 2
        {'f_1': (        # Yes depth 1
            0,                    # No depth 2
            {'f_11': (    # Yes depth 2
                0,                # No depth 3
                1                 # Yes depth 3
                )}
            ) # Depth 3 
        } # Depth 2
    ) # Depth 1
} 
DT = Tree()
DT.tree = tree
Y = np.array([1, 0, 0, 0, 1, 0, 1])
XX = np.array([
    [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0],
    [1, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]
])
# %%
DT.train(XX, Y)
# %%
DT._decs_accuracy(XX.T[2], Y)
# %%
