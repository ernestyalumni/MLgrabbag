"""
@file linreg_gradDes.py

@author Ernest Yeung
@email ernestyalumni dot gmail dot com

Linear regression, with (batch) gradient descent, implemented with Theano on GPU(s)

"""
import theano
import numpy as np
import theano.tensor as T
from theano import sandbox

class LinearReg(object):
	"""LinearReg Linear Regression Class
	
	@author Ernest Yeung
	@email ernestyalumni dot gmail dot com
	
	
	
	"""

	def __init__(self, d, m, alpha=0.01): 
		""" Initialize the parameters of the linear regression  

		@type d:  (positive) integer
		@param d: number of features 
			
		@type m: (positive) integer
		@param m : number of training examples, training data points, i.e. input data points

		@type alpha : float or double
		@param alpha : (constant, fixed) learning rate, also denoted by gamma by others

		"""
		#d d = features (number of)
		self.d = d	

		# Declare Theano symbolic variables
		self.X = T.matrix('X')
		self.y = T.vector('y')

		# Initializing "weights" or parameters theta
		theta_0 = np.zeros( d + 1 ); 
		self.theta = theano.shared( theta_0.astype("float32"), "theta") 
		
		# learning rate
		self.alpha = np.float32( alpha ) 
		
		# number of training data points
		self.m = m 
		
		# Construct Theano "expression graph", i.e. symbolic description of 
		# how to compute prediction, cost function J_theta
		self.predicted_vals = sandbox.cuda.basic_ops.gpu_from_host( 
			T.dot( self.X, self.theta) )  # this is linear regression model
	
		self.J_theta = sandbox.cuda.basic_ops.gpu_from_host( 
			T.dot( 
				(T.dot( self.X,self.theta) - self.y).T , 
				T.dot( self.X, self.theta) - self.y 
				) * np.float32( 0.5 * 1. / self.m ) 
				) # cost function
				
		self.update_theta = sandbox.cuda.basic_ops.gpu_from_host( 
			self.theta - self.alpha * T.grad( self.J_theta, self.theta ) )
			
		self.gradientDescent = theano.function( 
									inputs = [self.X, self.y ], 
									outputs=[self.predicted_vals, self.J_theta], 
									updates=[(self.theta, self.update_theta )],
									name = "gradientDescent" )
									
	def build_model(self, X, y, num_iters=1500, store_J = False):
		""" build_model - build actual mode
		
		@type X  : numpy array of dim. (m, d+1) i.e. (number of training example, number of features on training data X + 1 (for intercept)
		@param X : input training data, with column of 1's FIRST, for the intercept

		@type y  : numpy array of dim. (m,) i.e. (number of training examples, )
		@param y : test data y
		
		@type num_iters  : (positive) integer
		@param num_iters : number of iterations

		@type store_J    : boolean (default is False)
		@param store_J   : True if we want to store the value of cost function J; keep False to save computation time
		"""
		J_History = np.zeros(num_iters) 
		for iter in range(num_iters):
			predicted_vals_out, J_out = self.gradientDescent( X.astype("float32"), y.astype("float32") )
			
			if store_J:
				J_History[iter] = J_out.get_value()
			
		return self.theta.get_value()
		 
	def preprocess_X(self,X):
		""" preprocess_X preprocess X input training data by adding a column of 1's FIRST for the 
		intercept
		"""
		X_out = np.hstack( ( np.ones((self.m,1)), X) ).astype("float32")
		return X_out
		 
		 
class LinearReg_loaded(object):
	"""LinearReg_loaded Linear Regression Class, but with the input data already loaded
	
	"""
	def preprocess_X(self,X,m):
		""" preprocess_X preprocess X input training data by adding a column of 1's FIRST for the 
		intercept

		@type X  : numpy array of dim. (m,d) i.e. (number of training example, number of features on training data  
		@param X : input training data
		
		@type m  : (positive) integer
		@param m : number of (input) training data (points)
		"""
		X_out = np.hstack( ( np.ones((m,1)), X) ).astype("float32")
		return X_out

	
	def __init__(self, X,y, d, m , alpha= 0.01 ): 
		""" Initialize the parameters of the linear regression
		
		@type X  : numpy array of dim. (m, d) i.e. (number of training example, number of features on training data X
		@param X : input training data, with column of 1's FIRST, for the intercept

		@type y  : numpy array of dim. (m,) i.e. (number of training examples, )
		@param y : test data y

		@type m  : (positive) integer
		@param m : number of (input) training data (points)

		"""

		#d d = features (number of)
		self.d = d

		# number of training data points
		self.m = m 

		# hstack or concatenate a column of 1s first for the intercept
		processed_X = self.preprocess_X(X, self.m)

		self.X = theano.shared( processed_X.astype('float32'), name='X')
		self.y = theano.shared( y.astype('float32'), name='y')

		# Initializing "weights" or parameters theta
		theta_0 = np.zeros( d + 1 ); 
		theta_0 = np.vstack( theta_0).astype("float32")
		
		self.theta = theano.shared(  theta_0 , name='theta')
		
		
		# learning rate
		self.alpha = np.float32( alpha ) 
		
		# Construct Theano "expression graph"
		self.predicted_vals = sandbox.cuda.basic_ops.gpu_from_host( 
			T.dot(self.X, self.theta) ) # h_{\theta}
			
		# cost function J_theta, J_{\theta}
		self.J_theta = sandbox.cuda.basic_ops.gpu_from_host( 
				(
					T.dot( (T.dot(self.X,self.theta) - self.y).T, T.dot(self.X, self.theta)-self.y) 
						* np.float32( 0.5 / m ) 
				).reshape([] )
		) # cost function # reshape is to force "broadcast" into 0-dim. scalar for cost function J
				
		self.update_theta = sandbox.cuda.basic_ops.gpu_from_host( 
			self.theta - self.alpha * T.grad( self.J_theta, self.theta) ) 
			
		# Note that we removed the input values because we will always use the same shared variable
		# GPU Note: Removed the input values to avoid copying data to the GPU
		
		self.gradientDescent = theano.function( 
											inputs = [],
					#						outputs=[predicted_vals,J_theta], 						
											updates = [(self.theta,self.update_theta)],
											name = "gradientDescent" )
		
		self.calculate_cost = theano.function( [], self.J_theta )
		self.predict        = theano.function( [], predicted_vals )
		
	def build_model(self,num_iters=1500):
		""" build_model Build the linear regression model
		
		"""
		
		for iter in range(num_iters):
			self.gradientDescent()
			
		return self.theta.get_value()
		
	

		
