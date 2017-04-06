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

class Thetab(object):
	""" Thetab - Thetab class for parameters or weights and intercepts b between layers l and l+1

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

	@type .g      : function, could be theano function
	@param .g     : function (e.g. such as sigmoid, tanh, softmax, etc.)

	NOTES
	=====
	borrow=True GPU is ok
	cf. http://deeplearning.net/software/theano/tutorial/aliasing.html

	after initialization (automatic, via __init__), remember to "connect the layers l and l+1" with the class method connect_through
	"""

	def __init__(self, rng, l, s_ls, al=None, Theta=None, b=None, activation=T.tanh):
		""" Initialize the parameters for the `layer`

		@type rng  : numpy.random.RandomState
		@param rng : random number generator used to initialize weights
		
		@type l    : (positive) integer
		@param l   : layer number label, l=1 (input),2,...L-1, L is "output layer"

		@type s_ls : tuple of (positive) integers of size (length) 2, only
		@param s_ls : "matrix" size dimensions of Theta or weight mmatrix, of size dims. (s_lp1, s_l) (this is important)
		Bottom line: s_ls = (s_lp1, s_l)

		@type al     : theano shared variable or vector of size dims. (s_l, m), m=1,2..., number of training examples
		@oaram al    : "nodes" or "units" of layer l

		@type Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l+1}x(s_l)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param Theta : "weights" or parameters for l, l=1,2, ... L-1
		
		@type b      : theano shared of dim. s_{l+1} or s_lp1
		@param b     : intercepts

		@type activation  : theano.Op or function
		@param activation : Non linearity to be applied in the layer(s)
		"""
		s_lp1, s_l = s_ls

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
#			al_values =  np.array( np.zeros( (s_l,m)) , dtype=theano.config.floatX)
#			al = theano.shared(value=al_values, name='al', borrow=True)
			al = T.matrix()
			
			
		self.Theta = Theta  # size dims. (s_l,s_lp1) i.e. s_lp1 x s_l
		self.b     = b      # dims. s_lp1
		self.al    = al     # dims. s_l
		
		self.l     = l

		if activation is None:
			self.g = None
		else:
			self.g = activation
	

	def connect_through(self):
		""" connect_through

			Note that I made connect_through a separate class method, separate from the automatic initialization, 
			because you can then make changes to the "layer units" or "nodes" before "connecting the layers"
				"""
		lin_zlp1 = T.dot( self.Theta, self.al)+T.tile(self.b, (1,self.al.shape[1].astype('int32') ) ) # z^{(l+1)}
		if self.g is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.g( lin_zlp1 )

class MLP(object):
	""" MLP - Multilayer perceptron 
	
	The purpose of the MLP class is to demonstrate how to use the classes and functions here to go 
	from input data all the way to a trained model, predicted values, and accuracy test.   
	"""
	def __init__(self, L, s_l, X, y, activation_fxn=T.tanh, lambda_val=1.):
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
		
#		@type m : (positive) integer
#		@param m : total number of training examples
		
		@type X : numpy array of size dims. (d,m)
		@param X : input data to train on.  NOTE the size dims. or shape and how it should equal what's inputted in d,m
		
		@type y : numpy array of size dims. (K,m)
		@param y : output data of training examples
		
		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		
	
		"""
		self.L = L 
		# self.d = d (number of features in input data) assert s_l[0] == d True
#		self.m = m 
		
		self.s_l = s_l
		
		self.X=X
		self.y=y
		
		rng=np.random.RandomState(1234)
		
		###############
		# BUILD MODEL #
		###############

		# symbolic Theano variable 
		self._X_in_sym = T.matrix() # X input data represented as a symbolic variable 
		# symbolic Theano variable 
		self._y_in_sym = T.matrix() # y output data represented as a symbolic variable


		Thetabs_lst = [] # temporary list of Thetas, Theta,b weights or parameters

		Thetab1 = Thetab(rng,1,(s_l[1],s_l[0]),al=self._X_in_sym, activation=activation_fxn)
		Thetab1.connect_through()
		
		Thetabs_lst.append( Thetab1 )
		for l in range(2,L):
			inputlayer_al = Thetabs_lst[-1].alp1
			Thetabl = Thetab(rng,l,(s_l[l],s_l[l-1]),al=inputlayer_al, activation=activation_fxn)
			Thetabl.connect_through()
			Thetabs_lst.append( Thetabl )
		self.Thetabs = Thetabs_lst
		
		h_Theta = self.Thetabs[-1].alp1 # output layer, which should be predicted values according to hypothesis h_Theta
		
		Thetas_lst = [Thetb.Theta for Thetb in self.Thetabs ]
		bs_lst     = [Thetb.b for Thetb in self.Thetabs]
		
		self.J_Theta = build_cost_functional( L, lambda_val, h_Theta, self._y_in_sym, Thetas_lst) 

	###############################
	# BUILD MODEL('S UPDATE STEP) #
	###############################

	def build_update(self, X_vals = None, y_vals=None, alpha=0.01, beta=0.0):
		"""
		@type alpha : float
		@param alpha : learning rate
		
		@type beta : float
		@param beta : "momentum" parameter (it's in the update step for gradient descent)

		"""
		if X_vals is None:
			X_vals = self.X
		if y_vals is None:
			y_vals = self.y
		
		Thetas_lst = [Thetb.Theta for Thetb in self.Thetabs ]
		bs_lst     = [Thetb.b for Thetb in self.Thetabs]
		
		self.updateExpression, self.gradDescent_step = build_gradDescent_step( 
								self.J_Theta, Thetas_lst, bs_lst, 
								self._X_in_sym, self._y_in_sym, X_vals, y_vals,alpha,beta)

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
	def predicted_vals_logreg(self, X_vals=None ):
		""" build_predicted_vals_logreg - basically gives back h_{\theta}
		"""
		if X_vals is None:
			X_vals = self.X

		h_Theta = theano.function([],
#we're not doing any computing intensive gradient steps, just linear algebra once, so we don't need the gpu
#								outputs=sandbox.cuda.basic_ops.gpu_from_host( 
								outputs= self.Thetabs[-1].alp1,
									givens={ self._X_in_sym : X_vals.astype(theano.config.floatX) })
											
		return h_Theta()						


	# ACCURACY TESTS
	
	# for logistic regression on multiple classes K (so y is discrete, i.e. y\in \lbrace 1,2,..K\rbrace^m

	def accuracy_logreg(self,X_vals,y_vals):
		""" accuracy_logreg : accuracy score for logistic regression
		"""
		predicted_y = self.predicted_vals_logreg(X_vals) # size dims. (K,m) where K=s_L
		
		# put y data into y \in \lbrace 0,1,..K-1 \rbrace^m form, as opposed to y \in ( \lbrace 0 , 1 \rbrace^K )^m
		y_cls = np.vstack( np.argmax( y_vals ,axis=0) ) # size dims. (m,)-> (m,1)
		
		# put predicted y values into y\in\lbrace 0,1,...K-1 \rbrace^m form; remember to add 1 to change from Python index counting from 0,1,...K-1 to k=1,2,...K
		predicted_y_cls = np.vstack(np.argmax( predicted_y, axis=0) )
	
		accuracy_score = np.mean( predicted_y_cls == y_cls )
		print("Accuracy score : %.6f " % accuracy_score )
		return accuracy_score
	
# EY - 20170218		
# Programming/Concept note - is the Universal Approximator more general than the Neural network classifier or vice versa?
# Who should inherit, in an object-oriented manner, from who?  
class Universal_Approx(object):
	""" Universal_Approx - The Universal Approximator 
	
	The purpose of the Universal Approximator class is to implement the universal approximator described by
	Kurt Hornik (1991), 
	from input data all the way to a trained model, predicted values, and accuracy test.   

	cf. Kurt Hornik.  "Approximation Capabilities of Multilayer Feedforward Networks."  Neural Networks, Vol. 4, pp. 251-257.  1991
	"""
	def __init__(self, L, s_l, X, y, activation_fxn=T.tanh, lambda_val=1.):
		""" Initialize MLP class

		INPUT/PARAMETER(S)
		==================
		@type L : (positive) integer
		@param L : total number of layers L, counting l=1 (input layer), and l=L (output layer), so only 1 hidden layer means L=3

		@type s_l : Python list (or tuple, or iterable) of (positive) integers
		@param s_l : list (or tuple or iterable) of length L, containing (positive) integers for s_l, size or number of "nodes" or "units" in layer l=1,2,...L; because of how Python counts from 0, it's indexed 0,1,...L-1
					NOTE that number of "components" of y, K, must be equal to s_L, s_L=K
						
		@type X : numpy array of size dims. (d,m)
		@param X : input data to train on.  NOTE the size dims. or shape and how it should equal what's inputted in d,m
		
		@type y : numpy array of size dims. (K,m)
		@param y : output data of training examples
		
		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		
	
		"""
		self.L = L 
		
		self.s_l = s_l
		
		self.X=X
		self.y=y
		
		rng=np.random.RandomState(1234)
		
		###############
		# BUILD MODEL #
		###############

		# symbolic Theano variable 
		self._X_in_sym = T.matrix() # X input data represented as a symbolic variable 
		# symbolic Theano variable 
		self._y_in_sym = T.matrix() # y output data represented as a symbolic variable


		Thetabs_lst = [] # temporary list of Thetas, Theta,b weights or parameters

		Thetab1 = Thetab(rng,1,(s_l[1],s_l[0]),al=self._X_in_sym, activation=activation_fxn)
		Thetab1.connect_through()
		
		Thetabs_lst.append( Thetab1 )
		for l in range(2,L-1):  # don't include the Theta,b going to the output layer in this loop
			inputlayer_al = Thetabs_lst[-1].alp1
			Thetabl = Thetab(rng,l,(s_l[l],s_l[l-1]),al=inputlayer_al, activation=activation_fxn)
			Thetabl.connect_through()
			Thetabs_lst.append( Thetabl )

		# (Theta,b), going to output layer
		inputlayer_al = Thetabs_lst[-1].alp1
		Thetabl = Thetab(rng,L-1,(s_l[L-1],s_l[L-2]),al=inputlayer_al, activation=None)
		Thetabl.connect_through()
		Thetabs_lst.append(Thetabl)

		self.Thetabs = Thetabs_lst
		
		h_Theta = self.Thetabs[-1].alp1 # output layer, which should be predicted values according to hypothesis h_Theta
		
		Thetas_lst = [Thetb.Theta for Thetb in self.Thetabs ]
		bs_lst     = [Thetb.b for Thetb in self.Thetabs]
		
		self.J_Theta = build_cost_functional( L, lambda_val, h_Theta, self._y_in_sym, Thetas_lst) 

	###############################
	# BUILD MODEL('S UPDATE STEP) #
	###############################

	def build_update(self, X_vals = None, y_vals=None, alpha=0.01, beta=0.0):
		"""
		@type alpha : float
		@param alpha : learning rate
		
		@type beta : float
		@param beta : "momentum" parameter (it's in the update step for gradient descent)

		"""
		if X_vals is None:
			X_vals = self.X
		if y_vals is None:
			y_vals = self.y
		
		Thetas_lst = [Thetb.Theta for Thetb in self.Thetabs ]
		bs_lst     = [Thetb.b for Thetb in self.Thetabs]
		
		self.updateExpression, self.gradDescent_step = build_gradDescent_step( 
								self.J_Theta, Thetas_lst, bs_lst, 
								self._X_in_sym, self._y_in_sym, X_vals, y_vals,alpha,beta)

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
	def predicted_vals_logreg(self, X_vals=None ):
		""" build_predicted_vals_logreg - basically gives back h_{\theta}
		"""
		if X_vals is None:
			X_vals = self.X

		h_Theta = theano.function([],
#we're not doing any computing intensive gradient steps, just linear algebra once, so we don't need the gpu
#								outputs=sandbox.cuda.basic_ops.gpu_from_host( 
								outputs= self.Thetabs[-1].alp1,
									givens={ self._X_in_sym : X_vals.astype(theano.config.floatX) })
											
		return h_Theta()						


	# ACCURACY TESTS
	
	# for logistic regression on multiple classes K (so y is discrete, i.e. y\in \lbrace 1,2,..K\rbrace^m

	def accuracy_logreg(self,X_vals,y_vals):
		""" accuracy_logreg : accuracy score for logistic regression
		"""
		predicted_y = self.predicted_vals_logreg(X_vals) # size dims. (K,m) where K=s_L
		
		# put y data into y \in \lbrace 0,1,..K-1 \rbrace^m form, as opposed to y \in ( \lbrace 0 , 1 \rbrace^K )^m
		y_cls = np.vstack( np.argmax( y_vals ,axis=0) ) # size dims. (m,)-> (m,1)
		
		# put predicted y values into y\in\lbrace 0,1,...K-1 \rbrace^m form; remember to add 1 to change from Python index counting from 0,1,...K-1 to k=1,2,...K
		predicted_y_cls = np.vstack(np.argmax( predicted_y, axis=0) )
	
		accuracy_score = np.mean( predicted_y_cls == y_cls )
		print("Accuracy score : %.6f " % accuracy_score )
		return accuracy_score



#def build_cost_functional(L, lambda_val, h, y_sym, m_sym, Thetas):
def build_cost_functional(L, lambda_val, h, y_sym, Thetas):
	""" build_cost_functional (with regularization) J=J_y(Theta,b) # J\equiv J_y(\Theta,b), but now with 
	X,y being represented as theano symbolic variables first, before the actual numerical data values are given

	INPUT/PARAMETERS
	================
	@type L     : (positive) integer for (total) number of layers L
	@param L    : number of layers L
	
#	@type y_in  : numpy array of size dims. (K,m), where K=number of "classes", m=number of training examples 
#	@param y_in : output data 
	
	@type y_sym  : theano symbolic matrix, such as T.matrix() 
	@param y_sym : output data as a symbolic theano variable
NOTE: y_sym = T.matrix(); # this could be a vector, but I can keep y to be "general" in size dimensions
	
	@type m_sym : theano symbolic scalar, i.e. T.scalar(theano.config.floatX)
	@param m_sym : it's necessary to represent m=number of training examples as a symbol as it's not known yet
	
	@type h     : theano shared variable of size dims. (K,m)
	@param h    : hypothesis

	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the layers l=1,2,...L-1
	NOTE: remember, we want a list of theano MATRICES, themselves, not the class

	"""
	m = y_sym.shape[1].astype(theano.config.floatX)
#	m = y_sym.shape[1].astype("int32") # AMAZINGLY, the size dimension of a theano symbolic variable, is a symbolic variable itself (!!!)
# this doesn't work yet; obtain this error: ValueError: setting an array element with a sequence.
	
	# logistic regression cost function J, with no regularization (yet)
	J_theta = T.mean( T.sum(
			- y_sym * T.log(h) - (np.float32(1)-y_sym) * T.log( np.float32(1) - h), axis=0), axis=0)
	
	reg_term = np.float32(lambda_val/ (2. )) /m *T.sum( [ T.sum( Theta*Theta) for Theta in Thetas] )
#	reg_term = np.float32(lambda_val/ (2. ))  * T.sum( [ T.sum( Theta*Theta) for Theta in Thetas] )

	J_theta = sandbox.cuda.basic_ops.gpu_from_host( J_theta + reg_term )
	return J_theta
	
	
def build_gradDescent_step( J, Thetas, bs, X_sym,y_sym, X_vals, y_vals,alpha =0.01, beta = 0.0):
	""" build_gradDescent_step - gradient Descent (with momentum), but from build_cost_functional for the J

	INPUT/PARAMETERS:
	=================

	@param J     : cost function (from build_cost_function)
	
	@type Thetas  : Python list (or tuple or iterable) of Thetas, weights matrices
	@param Thetas : 
	
	@type bs      : Python list (or tuple or iterable) of b's, vector of "weight" or parameter "intercepts" 
	
	@type X_sym   : theano symbolic variable, such as T.matrix()
	@param X_sym  : theano symbolic variable representing input X data 
	
	@type y_sym   : theano symbolic variable, such as T.matrix()
	@param y_sym  : theano symbolic variable representing output y data, outcomes
	
	@param X_vals : 
	
	@param y_vals :
	
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
											updates=zip(Theta_bs,updateTheta_bs),
											givens={ X_sym : X_vals.astype(theano.config.floatX), 
													y_sym : y_vals.astype(theano.config.floatX) },
											name="gradDescent_step")
	return updateTheta_bs, gradientDescent_step
		
	




if __name__ == "__main__":
	print("In main.")
	rng_test=  np.random.RandomState(1234)
#	test_layer = Layer( rng_test, 1, 784, 392, 5000 )
#	passthru = theano.function([], sandbox.cuda.basic_ops.gpu_from_host( test_layer.alp1 ) )
#	print( passthru.maker.fgraph.toposort() )
#	print( passthru().shape )
	X_in_sym = T.matrix(theano.config.floatX)

	Thetab1 = Thetab(rng_test, 1,(4,5), al=X_in_sym, activation=T.nnet.sigmoid)
	Thetab1.connect_through()
	Thetab2 = Thetab(rng_test,2,(3,4), al=Thetab1.alp1, activation=T.nnet.sigmoid)
	Thetab2.connect_through()
	y_in_sym = T.matrix(theano.config.floatX) # theano symbolic variable representing
	
	Theta12_lst = [ Thetab1.Theta, Thetab2.Theta]
	b12_lst = [ Thetab1.b, Thetab2.b ]
#	m_sym    = T.scalar()
#	J = build_cost_functional(2, 1., Thetab2.alp1, y_in_sym, m_sym, [Thetab1.Theta, Thetab2.Theta] )
#	J = build_cost_functional(2, 1., Thetab2.alp1, y_in_sym, [Thetab1.Theta, Thetab2.Theta] )
	J = build_cost_functional(2, 1., Thetab2.alp1, y_in_sym, Theta12_lst)


	test_X_vals = np.array( range(1,21)).reshape(4,5)
	test_y_vals = np.array( range(2,12)).reshape(2,5)
	
	test_build_update = build_gradDescent_step( J, Theta12_lst, b12_lst, 
							X_in_sym, y_in_sym, test_X_vals, test_y_vals)
	
	test_MLP12 = MLP(3,[4,3,2],test_X_vals,test_y_vals)
	test_MLP12.build_update()
