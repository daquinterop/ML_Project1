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
	      /      \	      /      \
	     /        \      /        \
	    YES       NO    NO     function_11
	                             /    \
	                            /      \
	                           /        \
	                          NO        YES
tree = {    
    function: {
        function_0: (
            1,
            0
        },
        function_1: {
            0,
            function_11: {
                0, 1
            }
        }
    }
}

To access to nested dicts the best way is saving any of the N-depth dicts to 
a tmp dict, so that we avoid using multiple indexing, which should not be 
optimum for a non-fixed depth
'''

class Tree(dict):
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = dict()

    def get_node(self, node_coords):
        '''
        node_coords: coordinates vector for the leaf. It's a list with the 1/0 decisions at
        any depth, for instance [1, 0, 1] means node 0: yes; node_01 1: no, node_010 2: yes
        '''
        tmp_dict = self.tree
        for depth, coord in enumerate(node_coords):
            tmp_dict = tmp_dict[coord]


def DT_train_binary(X,Y,max_depth=3):
    '''
    X: 2D ndarray, binary features
    Y: 1D ndarray, binary target
    max_depth: int, maximum depth of the tree. If max_depth=-1, there's no max depth
    '''
