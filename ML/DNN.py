"""
@file DNN.py
@brief Deep Neural network, for multiple axons

@author Ernest Yeung
@email ernestyalumni dot gmail dot com
"""  

import theano
import numpy as np
import theano.tensor as T
from theano import sandbox

from six.moves import cPickle

########################################################################
### Thetab_right class #################################################
########################################################################

class Axon(object):
	""" Axon - Axon class, for parameters or weights and intercepts b between layers l and l+1, for right action, 
	i.e. the matrix (action) multiplies from the right onto the vector (the module or i.e. vector space)

	(AVAILABLE) CLASS MEMBERS
	=========================
	@type .Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_lxs_{l+1}), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
	@param .Theta : "weights" or parameters for l, l=1,2, ... L-1
		
	@type .b 	  : theano shared of dim. s_{l+1} or s_lp1
	@param .b     : intercepts

	@type .al 	  : theano shared variable or vector of size dims. (m,s_l), m=1,2..., number of training examples
	@param .al    : "nodes" or "units" of layer l

	@type .alp1   : theano symbolic expression for array of size dims. (m,s_lp1)
	@param .alp1  : "nodes" or "units" of layer l+1

	@type .l      : (positive) integer l=0,1,2,..L-1, L+1 is total number of layers, and l=L is "output" layer
	@param .l     : which layer we're "starting from" for Theta^{(l)}, l=0,1,...L-1

	@type .psi      : function, could be theano function
	@param .psi     : function (e.g. such as sigmoid, tanh, softmax, etc.)


	NOTES
	=====
	borrow=True GPU is ok
	cf. http://deeplearning.net/software/theano/tutorial/aliasing.html

	after initialization (automatic, via __init__), remember to "connect the layers l and l+1" with the class method connect_through
	"""

	def __init__(self, l, s_ls, al=None, Theta=None, b=None, activation=T.tanh, rng=None ):
		""" Initialize the parameters for the `layer`

		@type rng  : numpy.random.RandomState
		@param rng : random number generator used to initialize weights
		
		@type l    : (positive) integer
		@param l   : layer number label, l=0 (input),1,...L-1, L is "output layer"

		@type s_ls : tuple of (positive) integers of size (length) 2, only
		@param s_ls : "matrix" size dimensions of Theta or weight mmatrix, of size dims. (s_l, s_lp1) (this is important)
							Bottom line: s_ls = (s_l, s_lp1)

		@type al     : theano shared variable or vector of size dims. (m, s_l), m=1,2..., number of training examples
		@oaram al    : "nodes" or "units" of layer l

		@type Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l}x(s_{l+1})), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param Theta : "weights" or parameters for l, l=1,2, ... L-1
		
		@type b      : theano shared of dim. s_{l+1} or s_lp1; it's now a "row" array of that length s_lp1
		@param b     : intercepts

		@type activation  : theano.Op or function
		@param activation : Non linearity to be applied in the layer(s)
		"""
		s_l, s_lp1 = s_ls

		if rng is None:
			rng = np.random.RandomState(1234)

		if Theta is None:
			Theta_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( s_l + s_lp1 )), 
					high=np.sqrt(6. / ( s_l + s_lp1 )), size=(s_l, s_lp1) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				Theta_values *= np.float32( 4 )
			
			Theta = theano.shared(value=Theta_values, name="Theta"+str(l), borrow=True)
		

		if b is None:
			b_values =  np.ones(s_lp1).astype(theano.config.floatX)
			b= theano.shared(value=b_values, name='b'+str(l), borrow=True)	
			
		if al is None:
			al = T.matrix(dtype=theano.config.floatX)
			
	
		self.Theta = Theta  # size dims. (s_l,s_lp1) i.e. s_l x s_lp1
		self.b     = b      # dims. s_lp1
		self.al    = al     # dims. s_l
		
		self.l     = l

		if activation is None:
			self.psi = None
		else:
			self.psi = activation



	def connect_through(self, al_in=None):
		""" connect_through

			Note that I made connect_through a separate class method, separate from the automatic initialization, 
			because you can then make changes to the "layer units" or "nodes" before "connecting the layers"
				"""

		if al_in is not None:
			self.al = al_in

		lin_zlp1 = T.dot( self.al, self.Theta) + self.b
		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )

		return self.alp1

	def __get_state__(self):
		""" __get_state__ 

		This method was necessary, because 
		this is how our Theta/Layer combo/class, Thetab_right, INTERFACES with Feedforward

		OUTPUT(S)/RETURNS
		=================
		@type : Python dictionary 
		"""  
		Theta = self.Theta
		b = self.b
				
		Thetas = [ Theta, ]
		bs     = [ b, ]
		params = [ Theta, b]
		
		return dict(Thetas=Thetas,bs=bs,params=params)
		
	def __set_state__(self,*args):
		""" __set_state__
		"""
		Theta_in = args[0]
		b_in     = args[1]
		self.Theta.set_value(Theta_in)
		self.b.set_value(b_in)

########################################################################
### END of Thetab_right class ##########################################
########################################################################

########################################################################
## Feedforward_y_right class ###########################################
########################################################################


class Feedforward(object):
	""" Feedforward - Feedforward
	"""
	def __init__(self, L, s_l, activation_fxn=T.tanh, psi_Lm1=T.tanh, rng=None ):
		""" Initialize MLP class

		INPUT/PARAMETER(S)
		==================
		@type L : (positive) integer
		@param L : total number of axons L, counting l=0 (input layer), and l=L (output layer), so only 1 hidden layer means L=2

		@type s_l : Python list (or tuple, or iterable) of (positive) integers
		@param s_l : list (or tuple or iterable) of length L+1, containing (positive) integers for s_l, size or number of "nodes" or "units" in layer l=0,1,...L; 
					NOTE that number of "components" of y, K, must be equal to s_L, s_L=K
				
		"""
		self.L = L 
		self.s_l = s_l
	
		if rng is None:
			rng = np.random.RandomState(1234)
	
		###############
		# BUILD MODEL #
		###############

		Axons_lst = [] # temporary list of Thetas, Theta,b weights or parameters

		# initialize an instance of class Axon
		Axon0 = Axon(0, (s_l[0],s_l[1]), activation=activation_fxn, rng=rng)
		Axon0.connect_through()
		
		Axons_lst.append( Axon0 )
		
		for l in range(1,L-1): # don't include the Theta,b going to the output layer in this loop
			inputlayer_al = Axons_lst[-1].alp1

			#initialize an instance of class Axon
			Axonl = Axon(l,(s_l[l],s_l[l+1]),al=inputlayer_al, activation=activation_fxn, rng=rng)
			Axonl.connect_through()
			Axons_lst.append( Axonl )

		# (Theta,b), going to output layer, l=L
		if (L>1):
			inputlayer_al = Axons_lst[-1].alp1

			#initialize an instance of class Thetab_right
			Axonl = Axon(L-1,(s_l[L-1],s_l[L]),al=inputlayer_al, activation=psi_Lm1, rng=rng)
			Axonl.connect_through()
			Axons_lst.append(Axonl)
		
		self.Axons = Axons_lst
		

	def connect_through(self, X_in):
		""" connect_through - connect through the layers (the actual feedforward operation)

		INPUTS/PARAMETERS
		=================
		@type X_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		"""
		self.Axons[0].al = X_in
		self.Axons[0].connect_through()
		
		L = self.L
		
		for l in range(1,L):  	# l=1,2,...L-1, for each of the Theta operations between layers l
			self.Axons[l].al = self.Axons[l-1].alp1
			self.Axons[l].connect_through()	
		
		# return the result of Feedforward operation, denoted h_Theta
		h_Theta = self.Axons[-1].alp1
		return h_Theta

	def __get_state__(self):
		""" __get_state__ - return the parameters or "weights" that were used in this feedforward
		
		"""
		Axons = self.Axons

		Thetas = [theta for Axon in Axons for theta in Axon.__get_state__()['Thetas'] ]
		bs     = [b for Axon in Axons for b in Axon.__get_state__()['bs'] ]
		params = [weight for Axon in Axons for weight in Axon.__get_state__()['params']]

		
		return dict(Thetas=Thetas,bs=bs,params=params)
		
	def __set_state__(self,*args):
		""" __set_state__
		
		@type *args : expect a flattened out Python list of Axon* classes
		@param *args : use the "power" of *args so we're flexible about Python function argument inputting (could be a list, could be separate entries)
		"""
		L = self.L
		number_of_weights_1 = len(self.Axons[0].__get_state__()["params"]) # number of weights related to the 1st Thetab

		Axonl_vals=[args[w] for w in range(number_of_weights_1)]				

		self.Axons[0].__set_state__(*Axon1_vals)

		flattened_index = number_of_weights_1-1 # count from this value
		for l in range(1,L):  # l=1,...L-1 corresponds to index idx=1,...L-1 (Python counts from 0;I know, it's confusing)
			number_of_weightsl = len(self.Thetabs[l].__get_state__()['params'])# number of weights in layer l
			Axon_vals=[]
			for w in range(number_of_weightsl):
				Axon_vals.append( args[w+1+flattened_index] )
			
			self.Axons[l].__set_state__(*Axon_vals)
			flattened_index += number_of_weightsl
	
	def _get_outer_layer_(self):
		""" _get_outer_layer_ - return the theano (graph, computational) expression for the outer layer.  
								It's up to you to make a theano function, input this outer_layer expression for the outputs, and 
								compute a value
		"""
		Axons = self.Axons
		outer_layer = Axons[-1].alp1
		return outer_layer						

########################################################################
## END of Feedforward_y_right class ####################################
########################################################################

########################################################################
##################### Deep Neural Network DNN class ####################
########################################################################

class DNN(object):
	""" DNN - Deep Neural Network
	
	(AVAILABLE) CLASS MEMBERS
	=========================
	
	"""
	def __init__(self, DNN_model, X=None, y=None, lambda_val=1. ):
		""" Initialize MemoryBlock class
		
		INPUT/PARAMETER(S)
		==================  

		@type X : numpy array of size dims. (m,d)
		@param X : input data to train on.  NOTE the size dims. or shape and how it should equal what's inputted in d,m
		
		@type y : numpy array of size dims. (m,K)
		@param y : output data of training examples

		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		
		"""
		self._DNN_model = DNN_model

		if X is None:
			X = T.matrix(dtype=theano.config.floatX)

		self.X = theano.shared( X.astype(theano.config.floatX))

		if y is None:
			y = T.matrix(dtype=theano.config.floatX)

		self.y = theano.shared( y.astype(theano.config.floatX))

		self.lambda_val = lambda_val

	def connect_through(self,X=None):
		""" connect_through - connect through the layers (the actual feedforward operation)

		"""
		if X is not None:
			self.X = theano.shared( X.astype(theano.config.floatX))
		X = self.X
		
#		DNN_model=self._DNN_model
		h = self._DNN_model.connect_through(X)
		return h  
		
	def build_cost_functional(self,y=None):
		""" build_cost_functional - build the cost functional
		"""
		if y is not None:
			self.y=theano.shared(y.astype(theano.config.floatX))
		Y=self.y
		h_Theta = self._DNN_model._get_outer_layer_()
		lambda_val=self.lambda_val
		
		Thetas = self._DNN_model.__get_state__()['Thetas']

		self.J_Theta=build_cost_functional(lambda_val,h_Theta,Y,Thetas)
		return self.J_Theta

	def build_cost_functional_L2norm(self,y=None):
		""" build_cost_functional - build the cost functional
		"""
		if y is not None:
			self.y=theano.shared(y.astype(theano.config.floatX))
		Y=self.y
		h_Theta = self._DNN_model._get_outer_layer_()
		lambda_val=self.lambda_val
		
		Thetas = self._DNN_model.__get_state__()['Thetas']

		self.J_Theta=build_cost_functional_L2norm(lambda_val,h_Theta,Y,Thetas)
		return self.J_Theta

		
	###############################
	# BUILD MODEL('S UPDATE STEP) #
	###############################	

	def build_update(self,alpha=0.01,beta=0.0,X=None,y=None):
		"""
		@type alpha : float
		@param alpha : learning rate
		
		@type beta : float
		@param beta : "momentum" parameter (it's in the update step for gradient descent)

		"""
		J=self.J_Theta
		
		if X is not None:
			self.X = theano.shared(X.astype(theano.config.floatX))
		X=self.X
		if y is not None:
			self.y = theano.shared(y.astype(theano.config.floatX))
		y=self.y

		Thetas_lst = self._DNN_model.__get_state__()['Thetas']
		bs_lst     = self._DNN_model.__get_state__()['bs']

		self.updateExpression, self.gradDescent_step=build_gradDescent_step(J,Thetas_lst,bs_lst,X,y,alpha,beta)

	###############
	# TRAIN MODEL #
	###############	
	
	def train_model(self, max_iters=1500):
		for iter in range( max_iters) :
			self.gradDescent_step()

	####################
	# Predicted values #
	####################

	def predict_on_X(self,X): 
		if X is not None:
			self.X = theano.shared(X.astype(theano.config.floatX))
		self.connect_through()
		h_Theta=theano.function([],outputs=self._DNN_model._get_outer_layer_())()
		return h_Theta	


def build_cost_functional(lambda_val, h, y_sh, Thetas):
	""" build_cost_functional (with regularization) J=J_y(Theta,b) # J\equiv J_y(\Theta,b), but now with 
	X,y being represented as theano symbolic variables first, before the actual numerical data values are given

	INPUT/PARAMETERS
	================	
	@type y_sh  : theano shared variable  
	@param y_sh : output data as a theano shared variable
	
	@type h     : theano shared variable of size dims. (m,K)
	@param h    : hypothesis

	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the axons l=0,1,...L-1
	NOTE: remember, we want a list of theano MATRICES, themselves, not the class

	"""
	m = y_sh.shape[1].astype(theano.config.floatX)
	J_theta = T.mean( T.sum(
			- y_sh * T.log(h) - (np.float32(1)-y_sh ) * T.log( np.float32(1) - h), axis=0), axis=0)

#	J_theta=T.sum(T.nnet.categorical_crossentropy(h,y_sh))
	
#	reg_term=np.float32(lambda_val/(2.))/m*T.sum([T.sum(Theta*Theta) for Theta in Thetas])
	reg_term = np.cast[theano.config.floatX](lambda_val/ (2. ))  *T.mean( [ T.sum( T.sqr(Theta), acc_dtype=theano.config.floatX) for Theta in Thetas], acc_dtype=theano.config.floatX )


	J_theta = sandbox.cuda.basic_ops.gpu_from_host( J_theta + reg_term )
	return J_theta

def build_cost_functional_L2norm(lambda_val,h,y_sym,Thetas):
	""" 
	build_cost_functional_L2norm (with regularization) J=J_y(Theta,b) # J\equiv J_y(\Theta,b), 
	for the L2 norm, or Euclidean space norm, but now with 
	X,y being represented as theano symbolic variables first, before the actual numerical data values are given

	INPUT/PARAMETERS
	================
	@type y_sym  : theano symbolic matrix, such as T.matrix() 
	@param y_sym : output data as a symbolic theano variable
NOTE: y_sym = T.matrix(); # this could be a vector, but I can keep y to be "general" in size dimensions
	
	@type h     : theano shared variable of size dims. (K,m) (size dim. might be (m,K) due to right action
	@param h    : hypothesis

	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the layers l=1,2,...L-1
	NOTE: remember, we want a list of theano MATRICES, themselves, not the class

	RETURN/OUTPUTS
	==============
	@type J_theta : theano symbolic expression

	"""

	J_theta = np.cast[theano.config.floatX](0.5) * T.mean( T.sqr(h - y_sym ))

	reg_term = np.cast[theano.config.floatX](lambda_val/ (2. ))  *T.mean( [ T.sum( T.sqr(Theta), acc_dtype=theano.config.floatX) for Theta in Thetas], acc_dtype=theano.config.floatX )


	J_theta = J_theta + reg_term
	return J_theta	




def build_gradDescent_step( J, Thetas, bs, X_sym,y_sym,X_vals,y_vals,alpha=0.01,beta=0.0):
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

#	gradientDescent_step = theano.function(inputs=[X_sym,y_sym],
#											outputs = J,
#											updates=zip(Theta_bs,updateTheta_bs),
#											name="gradDescent_step")
#	gradientDescent_step = theano.function([],
#											outputs = J,
#											updates=zip(Theta_bs,updateTheta_bs),
#											givens={X_sym:X_vals.astype(theano.config.floatX),
#													y_sym:y_vals.astype(theano.config.floatX)},
#											name="gradDescent_step")
	gradientDescent_step = theano.function([],
											outputs = J,
											updates=zip(Theta_bs,updateTheta_bs),
											name="gradDescent_step")


	return updateTheta_bs, gradientDescent_step




if __name__ == "__main__":
	print("In main")
	rng_test = np.random.RandomState(1234)
	
	m_test = 3 # test value for m, total number of examples
	d_test =  4 # test value for total number of features
	X_sh_test = theano.shared( np.arange(2,2+m_test*d_test).reshape(m_test,d_test).astype(theano.config.floatX )) # theano shared test variable
	Axon0 = Axon(0,(d_test,5), al=X_sh_test, activation=T.nnet.sigmoid, rng=rng_test )
	Axon0.connect_through()
	Axon0.__get_state__()
	test_h0 = theano.function(inputs=[],outputs=Axon0.alp1)
	print(test_h0())
	
	L_test=1
	s_ls_test = [d_test,2*d_test]
	linANN1 = Feedforward( L_test, s_ls_test)
	L_test=2
	s_ls_test = [d_test,2*d_test,2]
	ANN2 = Feedforward(2, s_ls_test)
	ANN2.connect_through(X_sh_test)
	test_h2 = theano.function(inputs=[],outputs=ANN2._get_outer_layer_() )
	print(test_h2())
	
	
	
	



