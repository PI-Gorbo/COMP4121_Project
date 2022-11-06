
"""

 - Goal : Define a set of parameters (p1, p2, ... , pn) which operate on the vector of features (x1, x2, ..., xn, 1)
 -      The dot product of the parameters, dot(params, features) represents a hyperplane that goes through the feature space.
 -      Where any value above the feature plane is categorized into one class, and any value under the feature plane is in the other class.
 - Note: Since we are dealing with points on a plane, it is a good idea to scale the data.
 - Note, our y_i values need to be either [-1 or 1]
 - How to find the hyperplane:
 -     Find the optimal values of (p1, p2, ... , pn) through minimising a cost function
 -     The cost function is given by: 1/2 * ||w||^2 + C * [(1 / N) * sum( max(0, 1 - y_i * (w * x_i ) ) )] where C is the learning rate.  
 -     Gradient of the cost function is :
 -           dJ(w) / d(w_i) = 1 / N * sum( w if max(0, 1 - y_i * (w * x_i)) < 0 otherwise w - C * y_i * x_i)
 -      Use graident decent to minimise the cost function
 -      Store the parameters used for predictions 

"""


class SvmModel:
    pass