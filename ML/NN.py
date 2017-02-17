"""
@file NN.py
@brief Neural net, for multiple layers

@author Ernest Yeung
@email ernestyalumni dot gmail dot com
"""  

import theano
import numpy as np
import theano.tensor as T
from theano import sandbox

class Layer(object):
	"""Layer Layer Class for a "layer", but which actually corresponds to Theta or "weights" between layers
	
	"""
	
	def __init__(self, rng, l , s_l, s_lp1, m, al=None, Theta=None, b=None,  activation=T.tanh): 
		""" Initialize the parameters for the `layer`
	
		@type rng  : numpy.random.RandomState
		@param rng : random number generator used to initialize weights
		
		@type l    : (positive) integer
		@param l   : layer number label, l=1 (input),2,...L-1, L is "output layer"

		@type s_l  : (positive) integer
		@param s_l : dimensionality of layer l, i.e. number of "nodes" or "units" in layer l
		
		@type s_lp1  : (positive) integer
		@param s_lp1 : dimensionality of layer l+1, i.e. number of "nodes" or "units" in layer l+1
		
		@type m      : (positive) integer
		@param m     : (total) number of training examples

		@type al     : theano shared variable or vector of size dims. (s_l, m), m=1,2..., number of training examples
		@oaram al    : "nodes" or "units" of layer l

		@type Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l+1}x(s_l)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param Theta : "weights" or parameters for l, l=1,2, ... L-1
		
		@type b      : theano shared of dim. s_{l+1} or s_lp1
		@param b     : intercepts
		
		
		@type activation  : theano.Op or function
		@param activation : Non linearity to be applied in the layer(s)

		(AVAILABLE) CLASS MEMBERS
		=========================
		@type .Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l+1}x(s_l)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param .Theta : "weights" or parameters for l, l=1,2, ... L-1
		
		@type .b 	  : theano shared of dim. s_{l+1} or s_lp1
		@param .b     : intercepts

		@type .al 	  : theano shared variable or vector of size dims. (s_l, m), m=1,2..., number of training examples
		@param .al    : "nodes" or "units" of layer l

		@type .alp1   : theano symbolic expression for array of size dims. (s_lp1,m)
		@param .alp1  : "nodes" or "units" of layer l+1

		@type .l      : (positive) integer l=1,2,..L-1, L is total number of layers, and l=L is "output" layer
		@param .l     : which layer we're "starting from" for Theta^{(l)}, l=1,2,...L-1

		NOTES
		=====
		borrow=True GPU is ok
		cf. http://deeplearning.net/software/theano/tutorial/aliasing.html

		"""
		
		if Theta is None:
			Theta_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( s_l + s_lp1 )), 
					high=np.sqrt(6. / ( s_l + s_lp1 )), size=(s_lp1, s_l) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				Theta_values *= np.float32( 4 )
			
					
			Theta = theano.shared(value=Theta_values, name="Theta"+str(l), borrow=True)

		if b is None:
			b_values = np.vstack( np.zeros(s_lp1)).astype(theano.config.floatX)
			b= theano.shared(value=b_values, name='b'+str(l), borrow=True)	
			
		if al is None:
			al_values =  np.array( np.zeros( (s_l,m)) , dtype=theano.config.floatX)
			al = theano.shared(value=al_values, name='al', borrow=True)
			
			
		self.Theta = Theta  # size dims. (s_l,s_lp1) i.e. s_lp1 x s_l
		self.b     = b      # dims. s_lp1
		self.al    = al     # dims. s_l
		
		self.l     = l
		
		# Forward propagation; feed forward propagation
		lin_zlp1 = T.dot( self.Theta, self.al ) + T.tile( self.b, (1,m) )
		if activation is None:
			self.alp1 = lin_zlp1 
			self.g = None

		else:
			self.alp1 = activation( lin_zlp1 )  
			self.g = activation


def cost_functional_noreg(y_in, h):
	""" cost_functional J==J_y(Theta,b) # J\equiv J_y(\Theta,b)
	
	INPUT/PARAMETERS
	================
	@type y_in  : numpy array of size dims. (K,m), where K=number of "classes", m=number of training examples 
	@param y_in : output data 
	
	@type h     : theano shared variable of size dims. (K,m)
	@param h    : hypothesis
	"""
	K = y_in.shape[0]
	m = y_in.shape[1]
	
	y = theano.shared( np.asarray( y_in, dtype=theano.config.floatX), name='y') # size dim. (K,m)
	
	# logistic regression cost function J, with no regularization
	J_theta = sandbox.cuda.basic_ops.gpu_from_host( 
		T.mean( T.sum( 
			- y * T.log( h ) - ( np.float32(1) - y ) * T.log( np.float32(1) - h), axis=0), axis=0)
		)

	return J_theta

""" =================================
# 	cost functional WITH REGULARIZATION
""" 
def cost_functional(L, lambda_val, y_in, h, Thetas):
	""" cost_functional (with regularization) J==J_y(Theta,b) # J\equiv J_y(\Theta,b)
	
	INPUT/PARAMETERS
	================
	@type L     : (positive) integer for (total) number of layers L
	@param L    : number of layers L
	
	@type y_in  : numpy array of size dims. (K,m), where K=number of "classes", m=number of training examples 
	@param y_in : output data 
	
	@type h     : theano shared variable of size dims. (K,m)
	@param h    : hypothesis

	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the layers l=1,2,...L-1

	"""
#K = y_in.shape[0]
	m = y_in.shape[1]
	
	y = theano.shared( np.asarray( y_in, dtype=theano.config.floatX), name='y') # size dim. (K,m)
	
	# logistic regression cost function J, with no regularization
	J_theta =  T.mean( T.sum( 
			- y * T.log( h ) - ( np.float32(1) - y ) * T.log( np.float32(1) - h), axis=0), axis=0)
		
	reg_term = np.float32(lambda_val/ (2. * m)) * T.sum( [ T.sum( Theta*Theta ) for Theta in Thetas] ) 
	J_theta = sandbox.cuda.basic_ops.gpu_from_host( J_theta + reg_term )

	return J_theta


def gradientDescent_step( J, Thetas, bs, alpha =0.01, beta = 0.0):
	""" gradientDescent_step - gradient Descent (with momentum)

	INPUT/PARAMETERS:
	=================
	
	@param J     : cost function
	
	@type Thetas  : Python list (or tuple or iterable) of Thetas, weights matrices
	@param Thetas : 
	
	@type bs      : Python list (or tuple or iterable) of b's, vector of "weight" or parameter "intercepts" 
	
	@param alpha : learning rate
	
	@param beta  : "momentum" constant (parameter)

	RETURN(S)
	=========
	@type updateThetas, gradientDescent_step : tuple of (list of theano symbolic expression, theano function)

	"""

	updateThetas = [ sandbox.cuda.basic_ops.gpu_from_host( 
					Theta - np.float32( alpha) * T.grad( J, Theta) + np.float32(beta)*Theta ) for Theta in Thetas]
	
	updatebs     = [ sandbox.cuda.basic_ops.gpu_from_host(
					b - np.float32(alpha) * T.grad( J, b) + np.float32(beta) * b ) for b in bs]

	Theta_bs = Thetas + bs # concatenate Python lists
	updateTheta_bs = updateThetas + updatebs
	
	gradientDescent_step = theano.function([], 
											updates=zip(Theta_bs, updateTheta_bs),
											name="gradientdescent_step" )

	return updateTheta_bs, gradientDescent_step


def predicted_logreg( h_Theta ):
	""" predict_logreg - predict values for logistic regression
	
	RETURN(S)
	=========
	@type : theano function
	"""
	predicted_logreg = theano.function([], h_Theta) 
	return predicted_logreg



class MLP(object):
	""" MLP - Multilayer perceptron 
	
	The purpose of the MLP class is to demonstrate how to use the classes and functions here to go 
	from input data all the way to a trained model, predicted values, and accuracy test.   
	"""
	def __init__(self, L, s_l, m, X, y, activation_fxn=T.tanh, lambda_val=1.,alpha=0.1,beta=0.0001 ):
		""" Initialize MLP class

		INPUT/PARAMETER(S)
		==================
		@type L : (positive) integer
		@param L : total number of layers L, counting l=1 (input layer), and l=L (output layer), so only 1 hidden layer means L=3

		@type s_l : Python list (or tuple, or iterable) of (positive) integers
		@param s_l : list (or tuple or iterable) of length L, containing (positive) integers for s_l, size or number of "nodes" or "units" in layer l=1,2,...L; because of how Python counts from 0, it's indexed 0,1,...L-1
					NOTE that number of "components" or classes of y, K, must be equal to s_L, s_L=K
						
#		@type d : (positive) integer
#		@param d : total number of features for input data
		
		@type m : (positive) integer
		@param m : total number of training examples
		
		@type X : numpy array of size dims. (d,m)
		@param X : input data to train on.  NOTE the size dims. or shape and how it should equal what's inputted in d,m
		
		@type y : numpy array of size dims. (K,m)
		@param y : output data of training examples
		
		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		
		@type alpha : float
		@param alpha : learning rate
		
		@type beta : float
		@param beta : "momentum" parameter (it's in the update step for gradient descent)
		

		"""
		self.L = L 
#		self.d = d  assert s_l[0] == d True
		self.m = m 
		# assert X.shape[0] == d, X.shape[1] == m

		self.s_l = s_l

		self.y = y

		rng = np.random.RandomState(1234)
		
		###############
		# BUILD MODEL #
		###############
#		self.Thetas = [Layer(rng,l,s_l[l-1],s_l[l],m,activation=activation_fxn) for l in range(1,L)]  # l=1,2,..L-1

		Thetas_lst = [] # temporary list of Thetas, Theta,b weights or parameters

		Thetas_lst.append( Layer(rng,1,s_l[0],s_l[1],m,activation=activation_fxn) )
		for l in range(2,L):
			inputlayer_al = Thetas_lst[-1].alp1
			Thetas_lst.append( Layer(rng,l,s_l[l-1],s_l[l],m,al=inputlayer_al, activation=activation_fxn))
		self.Thetas = Thetas_lst


		self.Thetas[0].al.set_value( X.astype(theano.config.floatX) )
		
		h_Theta = self.Thetas[-1].alp1 # output layer, which should be predicted values according to hypothesis h_Theta

		Thetas_lst = [Thet.Theta for Thet in self.Thetas ]
		bs_lst     = [Thet.b for Thet in self.Thetas ]
		self.J_Theta = cost_functional(L,lambda_val,y, h_Theta, Thetas_lst)
	
		# update (theano) EXPRESSION, actual theano function gradient descent step
		self.updateExpression, self.gradDescent_step = gradientDescent_step( self.J_Theta, 
															Thetas_lst, bs_lst, alpha, beta)
		

		
	###############
	# TRAIN MODEL #
	###############	
	
	def train_model(self, max_iters=1500):
		for iter in range( max_iters) :
			self.gradDescent_step()
			
	# Predicted values for logreg, logistic regression:
	# You can do this manually:
	# predicted_logreg( MLP.Thetas[-1].alp1 )
	# or like so
	def predicted_vals_logreg(self):
		predict_vals_func = predicted_logreg( self.Thetas[-1].alp1 )
		return predict_vals_func() # do the actual prediction on actual values, with the inputted X and trained Thetas,b's
	
	
	# ACCURACY TESTS
	
	# for logistic regression on multiple classes K (so y is discrete, i.e. y\in \lbrace 1,2,..K\rbrace^m
	def accuracy_log_reg(self):
		
		predicted_y = self.predicted_vals_logreg()  # size dims. (K,m) where K=s_L
		
		# put y data into y \in \lbrace 0,1,..K-1 \rbrace^m form, as opposed to y \in ( \lbrace 0 , 1 \rbrace^K )^m
		y_cls = np.vstack( np.argmax( self.y ,axis=0) ) # size dims. (m,)-> (m,1)
		
		# put predicted y values into y\in\lbrace 0,1,...K-1 \rbrace^m form; remember to add 1 to change from Python index counting from 0,1,...K-1 to k=1,2,...K
		predicted_y_cls = np.vstack(np.argmax( predicted_y, axis=0) )
	
		accuracy_score = np.mean( predicted_y_cls == y_cls )
		print("Accuracy score : %.6f " % accuracy_score )
		return accuracy_score

	
	# for PREDICTION (on validation examples, test examples)
	# define a theano symbolic variable to represent input data
		
	def predict_on(self, m_in, X_input):
		""" predict_on - use the current trained model (consisting of Theta,b 's) to make predictions, 
				given a new set of input data for X

		@type X_input : numpy array of size (d,m) where d=number of features, m=number of input data pts.
		@param X_input : input data X to make predictions on 

		RETURN(S)
		=========
		@type predict_givens : theano function
		@param predict_givens : theano function, when instantiated, gives a numpy array of predicted values 
		
		"""
		m = m_in # number of input data pts., assert X_input.shape[1] == m_in == m
		X_in = T.matrix()

		layer1 = self.Thetas[0]
		if layer1.g == None:
			alp1_given = T.dot( layer1.Theta, X_in ) + T.tile( layer1.b, (1,m))
		else:
			alp1_given = layer1.g( T.dot( layer1.Theta, X_in ) + T.tile( layer1.b, (1,m)) )

		for l_ind in range(1,self.L-1):  # l_in = 1,..L-2, corresponding, 1-to-1 to l=2,3,...L-1 (it's how Python counts from 0)
			layer = self.Thetas[l_ind]
			if layer.g == None:
				alp1_given = T.dot( layer.Theta, alp1_given) + T.tile( layer.b, (1,m))
			else:
				alp1_given = layer.g( T.dot( layer.Theta, alp1_given) + T.tile( layer.b, (1,m) ) )
				
		predict_givens = theano.function([], 
										outputs= alp1_given, 
							givens={ X_in: X_input.astype(theano.config.floatX) } ) 
		
		
		return predict_givens
	


if __name__ == "__main__":
	print("In main.")
	rng_test=  np.random.RandomState(1234)
	test_layer = Layer( rng_test, 1, 784, 392, 5000 )
	passthru = theano.function([], sandbox.cuda.basic_ops.gpu_from_host( test_layer.alp1 ) )
	print( passthru.maker.fgraph.toposort() )
	print( passthru().shape )
