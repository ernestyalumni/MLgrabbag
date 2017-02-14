"""
@file gradDes.py

@author Ernest Yeung
@email ernestyalumni dot gmail dot com

Linear regression, 
log regression
with (batch) gradient descent, and regularization implemented with Theano on GPU(s)

TIPS for RUNNING gradDes.py
	Just this script itself (stand-alone), at Command Prompt $ :
	
THEANO_FLAGS='mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0.10' python -i gradDes.py
	
(adjust cnmem=0.1 according to how much memory you want to use)

"""

import theano
import numpy as np
import theano.tensor as T
from theano import sandbox


class LogReg(object):
	"""Reg Regression Class for Linear or logistic regression

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
	
	
		
	def __init__(self, X,y, d, m , alpha= 0.01, lambda_val = 1. ): 
		""" Initialize the parameters of the linear regression
		
		@type X  : numpy array of dim. (m, d) i.e. (number of training example, number of features on training data X
		@param X : input training data, with column of 1's FIRST, for the intercept

		@type y  : numpy array of dim. (m,) i.e. (number of training examples, )
		@param y : test data y

		@type m  : (positive) integer
		@param m : number of (input) training data (points)

		@type alpha  : float
		@param alpha : learning rate
		
		@type lambda_val  : float 
		@param lambda_val : regularization constant

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
		self.z = sandbox.cuda.basic_ops.gpu_from_host( 
			T.dot(self.X, self.theta) ) 

		self.h_theta = T.nnet.sigmoid( self.z) 
		
		# cost function J_theta, J_{\theta}, no regularization 
		J_theta = ( -T.dot( self.y.T, T.log(self.h_theta) ) - T.dot( 
			( np.float32(1.)-self.y).T , np.log( np.float32(1.) - self.h_theta )) )/ np.float32(m)
			
			
		# regularized term
		reg_term = T.dot( self.theta[1:].T , self.theta[1:] ) * np.float32( lambda_val / (2. * m ) )

		
		# cost function J_theta, J_{\theta} with regularization
		self.J_theta = sandbox.cuda.basic_ops.gpu_from_host( 
																(J_theta + reg_term ).reshape([] ) )
		# cost function # reshape is to force "broadcast" into 0-dim. scalar for cost function J

		self.update_theta = sandbox.cuda.basic_ops.gpu_from_host( 
										self.theta - self.alpha * T.grad( self.J_theta, self.theta) ) 
			
		# Note that we removed the input values because we will always use the same shared variable
		# GPU Note: Removed the input values to avoid copying data to the GPU
		
		self.gradientDescent = theano.function( 
											inputs = [],
											updates = [(self.theta,self.update_theta)],
											name = "gradientDescent" )

		self.calculate_cost = theano.function( [], self.J_theta )
		self.predict        = theano.function( [], self.h_theta )



				
	def build_model(self,num_iters=1500):
		""" build_model Build the linear regression model
		
		"""
		
		for iter in range(num_iters):
			self.gradientDescent()
			
		return self.theta.get_value()		

	def set_theta(self, theta):
		""" set_theta Set the value of theta, parameters or "weights", manually
		
			note that this can be done also in the following manner:
			
			yournamefor_LogReg.theta.set_value( theta_t.astype('float32') )

			with theta_t being your numpy array you'd want to input, size dx1, i.e. \theta \in \mathbb{R}^d
		"""
		theta_in = theta.astype('float32')
		self.theta.set_value( theta_in )



"""
test_function - testing about the classes in main
"""
def test_function():
	""" test_function - test function for gradDes
	
	"""

	theta_t = np.vstack( np.array( [-2, -1, 1, 2]) )
	X_t = np.array( [i/10. for i in range(1,16)]).reshape((3,5)).T
	y_t = np.vstack( np.array( [1,0,1,0,1]))

	nofeatures = 3
	notrainingexamples = 5
	learningratealpha = 0.01
	lambda_regulconst = 3.
	
	logreg_test_args = [ X_t, y_t, nofeatures, notrainingexamples, learningratealpha , lambda_regulconst ]
	
	MulClsCls_digits = LogReg( *logreg_test_args)
	MulClsCls_digits.set_theta( theta_t)
	return MulClsCls_digits


if __name__ == "__main__":
	MulCls2_test = test_function()
	print( '\n Cost: ' )
	print( MulCls2_test.calculate_cost().view() ) # 
	print('\n Expected cost: 2.534819\n') 
	print("Gradients:\n")
	
	gradfunc = theano.function([],T.grad( MulCls2_test.J_theta, MulCls2_test.theta ) )
	print( gradfunc().view() )
	print( "\n Expected gradients: \n" )
	print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')
	
	
