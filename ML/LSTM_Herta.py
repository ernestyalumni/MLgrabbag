"""
@file LSTM_Herta.py
@brief LSTM - Long Short-Term Memory, based on Herta's implementation

@author Ernest Yeung
@email ernestyalumni dot gmail dot com

cf. http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php
"""  

import theano
import numpy as np
import theano.tensor as T
from theano import sandbox

from six.moves import cPickle

from collections import namedtuple

""" Gates - namedtuple (from collections Python library) for the various "gates" needed
	'g' - g stands for input (either computer science or electrical engineer vernacular term (???))
	'i' - input control gate, should be either 1 or 0 to let input in or not
	'f' - "FORGET" gate - do we forget or keep what's in memory? Correspondingly, should be 0 or 1 for the "output" value of forget gate
	'o' - output control gate, should be either 1 or 0 to let predicted value "out" or not
"""
Gates = namedtuple("Gates",['g','i','f','o'])

""" Psis - namedtuple (from collections Python library) for the various "activation" functions needed
	'g' - g stands for input (either computer science or electrical engineer vernacular term (???))
	'i' - input control gate, should be either 1 or 0 to let input in or not
	'f' - "FORGET" gate - do we forget or keep what's in memory? Correspondingly, should be 0 or 1 for the "output" value of forget gate
	'o' - output control gate, should be either 1 or 0 to let predicted value "out" or not
	'h' - h for hypothesis, what your hypothesis, based on this choice of \Theta, b, \theta, predicts for the output value y
"""
Psis = namedtuple("Psis",['g','i','f','o','h'])


##########################################################################################
## Dr. Christian Herta's LSTM
##########################################################################################

class Thetab_right(object):
	""" Thetab_right - Thetab class for parameters or weights and intercepts b between layers l and l+1, for right action, 
	i.e. the matrix (action) multiplies from the right onto the vector (the module or i.e. vector space)

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

	@type .psi      : function, could be theano function
	@param .psi     : function (e.g. such as sigmoid, tanh, softmax, etc.)

	@type .params  : list of Theano shared variables
	@param .params : represents all the weights

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
		@param l   : layer number label, l=1 (input),2,...L-1, L is "output layer"

		@type s_ls : tuple of (positive) integers of size (length) 2, only
		@param s_ls : "matrix" size dimensions of Theta or weight mmatrix, of size dims. (s_l, s_lp1) (this is important)
		Bottom line: s_ls = (s_l, s_lp1)

		@type al     : theano shared variable or vector of size dims. (s_l, m), m=1,2..., number of training examples
		@oaram al    : "nodes" or "units" of layer l

		@type Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l}x(s_{l+1})), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param Theta : "weights" or parameters for l, l=1,2, ... L-1
		
		#@type b      : theano shared of dim. s_{l+1} or s_lp1; it's now a "row" array of that length s_lp1
		#@param b     : intercepts

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
			b_values =  np.zeros(s_lp1).astype(theano.config.floatX)
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
	

	def connect_through(self):
		""" connect_through

			Note that I made connect_through a separate class method, separate from the automatic initialization, 
			because you can then make changes to the "layer units" or "nodes" before "connecting the layers"
				"""
# my attempt at left action version; T.tile made problems for scan 
#		lin_zlp1 = T.dot( self.Theta, self.al)+T.tile(self.b, (1,self.al.shape[1].astype('int32') ) ) # z^{(l+1)}
		lin_zlp1 = T.dot( self.al, self.Theta) + self.b
		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )



class Thetabtheta_right(object):
	""" Thetabtheta_right - Thetabtheta class for parameters or weights and intercepts b between layers 1 and 2 for LSTM,
	with right action, i.e. the matrix (action) multiplies from the right onto the vector (the module or i.e. vector space)

	(AVAILABLE) CLASS MEMBERS
	=========================
	@type .Theta  : theano shared; of size dims. (s_l,s_lp1), i.e. matrix dims. (s_{l}x(s_{l+1)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
	@param .Theta : "weights" or parameters for l
		
	@type .b 	  : theano shared of dim. s_{l+1} 
	@param .b     : intercepts

	@type .theta  : theano shared of size dim. (H,s_lp1) i.e. \theta \in \text{Mat}_{\mathbb{R}}(H,s_{lp1})
	@param .theta : weight 

	@type .al 	  : theano shared variable or vector of size dims. (m, s_l), m=1,2..., number of training examples
	@param .al    : "nodes" or "units" of layer l
	
	@type ._h  : theano matrix
	@param ._h : symbolic variable representing the "hidden layer units"
	
	@type .alp1   : theano symbolic expression for array of size dims. (m,slp1)
	@param .alp1  : "nodes" or "units" of layer l+1

	@type .l      : (positive) integer l=1,2,..L-1, L is total number of layers, and l=L is "output" layer; default is 1
	@param .l     : which layer we're "starting from" for Theta^{(l)}, l=1,2,...L-1

	@type .psi      : function, could be theano function
	@param .psi     : function (e.g. such as sigmoid, tanh, softmax, etc.)

	NOTES
	=====
	borrow=True GPU is ok
	cf. http://deeplearning.net/software/theano/tutorial/aliasing.html

	after initialization (automatic, via __init__), remember to "connect the layers l and l+1" with the class method connect_through
	"""

	def __init__(self, s_ls, H, al=None, _h=None, Theta=None, b=None, theta=None, activation=T.tanh, l=1, rng=None):
		""" Initialize the parameters for the `layer`
		
		@type s_ls : tuple of (positive) integers of size (length) 2, only
		@param s_ls : "matrix" size dimensions of Theta or weight mmatrix, of size dims. (s_l, s_lp1) (this is important)
		Bottom line: s_ls = (s_l, s_lp1)

		@type H    : (positive) integer, default is 1
		@param H   : number of components of hidden data

		@type al     : theano shared variable or vector of size dims. (m,s_l), m=1,2..., number of training examples
		@oaram al    : "nodes" or "units" of layer l

		@type Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l}x(s_{l+1})), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param Theta : "weights" or parameters for l, l=1,2, ... L-1
		
		@type b      : theano shared of dim. s_{l+1} or s_lp1, but as a "row" array
		@param b     : intercepts

		@type activation  : theano.Op or function
		@param activation : Non linearity to be applied in the layer(s)

		@type l    : (positive) integer, default is 1
		@param l   : layer number label, l=1 (input),2,...L-1, L is "output layer"

		@type rng  : numpy.random.RandomState
		@param rng : random number generator used to initialize weights

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

		if theta is None:
			theta_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( H + s_lp1 )), 
					high=np.sqrt(6. / ( H + s_lp1 )), size=(H, s_lp1) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				theta_values *= np.float32( 4 )
			
			theta = theano.shared(value=theta_values, name="theta"+str(l), borrow=True)


		if b is None:
			b_values = np.zeros(s_lp1).astype(theano.config.floatX)
			b= theano.shared(value=b_values, name='b'+str(l), borrow=True)
			
		if al is None:
			al = T.matrix(dtype=theano.config.floatX)
		
		if _h is None:
			_h = T.matrix(dtype=theano.config.floatX)	
			
		self.Theta = Theta  # size dims. (s_l,s_lp1) i.e. s_l x s_lp1
		self.b     = b      # dims. s_lp1
		self.theta = theta  # size dims. H x s_lp1 i.e. theta \in \text{Mat}_{\mathbb{R}}( H,s_lp1)
		self.al    = al     # dims. s_1
		self._h     = _h      # dims. H
		
		self.l     = l # l is the layer (index) that this starts off at
		self.H     = H  # number of units or "nodes" in "hidden layer units"

		if activation is None:
			self.psi = None
		else:
			self.psi = activation
	

	def connect_through(self):
		""" connect_through

			Note that I made connect_through a separate class method, separate from the automatic initialization, 
			because you can then make changes to the "layer units" or "nodes" before "connecting the layers"
				"""
# my attempt at left action version; T.tile made problems for scan 
#		lin_zlp1 = T.dot( self.Theta, 
#							self.al)+T.tile(self.b, 
#											(1,self.al.shape[1].astype('int32'))) + T.dot( self.theta,
#																							self.y) # z^{(l+1)}
		lin_zlp1 = T.dot( self.al, 
						self.Theta) + self.b + T.dot( self._h, self.theta)

		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )

""" ThetabthetaW - for gates i,f,o, the input, forget, and output gates, respectively
"""
class ThetabthetaW_right(object):
	""" ThetabthetaW_right - ThetabthetaW class for parameters or weights and intercepts b between layers l and l+1 for LSTM,
	with right action, i.e. the matrix (action) multiplies from the right onto the vector (the module or i.e. vector space)

	(AVAILABLE) CLASS MEMBERS
	=========================
	@type .Theta  : theano shared; of size dims. (s_l,s_lp1), i.e. matrix dims. (s_{l}x(s_{l+1)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
	@param .Theta : "weights" or parameters for l
		
	@type .b 	  : theano shared of dim. s_{l+1} 
	@param .b     : intercepts

	@type .theta  : theano shared of size dim. (H,s_lp1) i.e. \theta \in \text{Mat}_{\mathbb{R}}(H,s_{lp1})
	@param .theta : weight 

	@type .W  : theano shared of size dim. H, s_lp1, i.e. \theta \in \text{Mat}_{\mathbb{R}}(H,s_lp1)
	@param .W : weight, that multiplies with "cell memory" value from t-1 

	@type .al 	  : theano shared variable or vector of size dims. (m, s_l), m=1,2..., number of training examples
	@param .al    : "nodes" or "units" of layer l
	
	@type ._h  : theano matrix
	@param ._h : symbolic variable representing "hidden layer units" 

	@type ._c  : theano matrix
	@param ._c : symbolic variable representing "cell (memory) layer units" 

	@type .alp1   : theano symbolic expression for array of size dims. (m,slp1)
	@param .alp1  : "nodes" or "units" of layer l+1

	@type .l      : (positive) integer l=1,2,..L-1, L is total number of layers, and l=L is "output" layer; default is 1
	@param .l     : which layer we're "starting from" for Theta^{(l)}, l=1,2,...L-1

	@type .psi      : function, could be theano function
	@param .psi     : function (e.g. such as sigmoid, tanh, softmax, etc.)

	NOTES
	=====
	borrow=True GPU is ok
	cf. http://deeplearning.net/software/theano/tutorial/aliasing.html

	after initialization (automatic, via __init__), remember to "connect the layers l and l+1" with the class method connect_through
	"""

	def __init__(self, s_ls, H, al=None, _h=None, _c=None,Theta=None, b=None, theta=None, W=None, 
					activation=T.tanh, l=1, rng=None):
		""" Initialize the parameters for the `layer`
		
		@type s_ls : tuple of (positive) integers of size (length) 2, only
		@param s_ls : "matrix" size dimensions of Theta or weight mmatrix, of size dims. (s_l, s_lp1) (this is important)
		Bottom line: s_ls = (s_l, s_lp1)

		@type H    : (positive) integer, 
		@param H   : number of components of hidden data

		@type al     : theano shared variable or vector of size dims. (m,s_l), m=1,2..., number of training examples
		@oaram al    : "nodes" or "units" of layer l

		@type ._h  : theano matrix
		@param ._h : symbolic variable representing "hidden layer units" 

		@type ._c  : theano matrix
		@param ._c : symbolic variable representing "cell (memory) layer units" 

		@type Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l}x(s_{l+1})), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param Theta : "weights" or parameters for l, l=1,2, ... L-1
		
		@type b      : theano shared of dim. s_{l+1} or s_lp1, but as a "row" array
		@param b     : intercepts

		@type .theta  : theano shared of size dim. (H,s_lp1) i.e. \theta \in \text{Mat}_{\mathbb{R}}(H,s_{lp1})
		@param .theta : weight 
	
		@type .W  : theano shared of size dim. H, s_lp1, i.e. \theta \in \text{Mat}_{\mathbb{R}}(H,s_lp1)
		@param .W : weight, that multiplies with "cell memory" value from t-1 

		@type activation  : theano.Op or function
		@param activation : Non linearity to be applied in the layer(s)

		@type l    : (positive) integer, default is 1
		@param l   : layer number label, l=1 (input),2,...L-1, L is "output layer"

		@type rng  : numpy.random.RandomState
		@param rng : random number generator used to initialize weights

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

		if theta is None:
			theta_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( H + s_lp1 )), 
					high=np.sqrt(6. / ( H + s_lp1 )), size=(H, s_lp1) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				theta_values *= np.float32( 4 )
			
			theta = theano.shared(value=theta_values, name="theta"+str(l), borrow=True)


		if b is None:
			b_values = np.zeros(s_lp1).astype(theano.config.floatX)
			b= theano.shared(value=b_values, name='b'+str(l), borrow=True)	

		if W is None:
			W_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( H + s_lp1 )), 
					high=np.sqrt(6. / ( H + s_lp1 )), size=(H, s_lp1) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				W_values *= np.float32( 4 )
			
			W = theano.shared(value=W_values, name="W"+str(l), borrow=True)
			
		if al is None:
			al = T.matrix(dtype=theano.config.floatX)
		
		if _h is None:
			_h = T.matrix(dtype=theano.config.floatX)	

		if _c is None:
			_c = T.matrix(dtype=theano.config.floatX)	

			
		self.Theta = Theta  # size dims. (s_l,s_lp1) i.e. s_l x s_lp1
		self.b     = b      # dims. s_lp1
		self.theta = theta  # size dims. H x s_lp1 i.e. theta \in \text{Mat}_{\mathbb{R}}( H,s_lp1)
		self.W     = W      # size dims. H x s_lp1 
		self.al    = al     # dims. s_1
		self._h     = _h      # dims. H
		self._c    = _c
		self.l     = l  # l is the layer that "we start off with"
		self.H     =H   # number of "hidden layer units"

		if activation is None:
			self.psi = None
		else:
			self.psi = activation
	

	def connect_through(self):
		""" connect_through

			Note that I made connect_through a separate class method, separate from the automatic initialization, 
			because you can then make changes to the "layer units" or "nodes" before "connecting the layers"
				"""
# my attempt at left action version; T.tile made problems for scan 
#		lin_zlp1 = T.dot( self.Theta, 
#							self.al)+T.tile(self.b, 
#											(1,self.al.shape[1].astype('int32'))) + T.dot( self.theta,
#																							self.y) # z^{(l+1)}
		lin_zlp1 = T.dot( self.al, 
						self.Theta) + self.b + T.dot( self._h, 
													self.theta) + T.dot( self._c, self.W)

		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )
		

class Feedforward_g_right(object):
	""" Feedforward - Feedforward
	"""
	def __init__(self, L, s_l, H, activation_fxn=T.tanh, psi_Lm1=T.tanh, rng=None ):
		""" Initialize MLP class

		INPUT/PARAMETER(S)
		==================
		@type L : (positive) integer
		@param L : total number of layers L, counting l=1 (input layer), and l=L (output layer), so only 1 hidden layer means L=3

		@type s_l : Python list (or tuple, or iterable) of (positive) integers
		@param s_l : list (or tuple or iterable) of length L, containing (positive) integers for s_l, size or number of "nodes" or "units" in layer l=1,2,...L; because of how Python counts from 0, it's indexed 0,1,...L-1
					NOTE that number of "components" of y, K, must be equal to s_L, s_L=K
		
		@type H    : (positive) integer, 
		@param H   : number of components of hidden data				
		
		"""
		self.L = L 
		self.s_l = s_l
	
		if rng is None:
			rng = np.random.RandomState(1234)
	
		###############
		# BUILD MODEL #
		###############

		Thetabs_lst = [] # temporary list of Thetas, Theta,b weights or parameters

		# initialize an instance of class Thetabtheta_right
		Thetabtheta1 = Thetabtheta_right( (s_l[0],s_l[1]), H,activation=activation_fxn, l=1, rng=rng)
		Thetabtheta1.connect_through()
		
		Thetabs_lst.append( Thetabtheta1 )
		
		for l in range(2,L-1): # don't include the Theta,b going to the output layer in this loop
			inputlayer_al = Thetabs_lst[-1].alp1

			#initialize an instance of class Thetabtheta_right
			Thetabl = Thetabtheta_right(l,(s_l[l-1],s_l[l]),al=inputlayer_al, activation=activation_fxn, rng=rng)
			Thetabl.connect_through()
			Thetabs_lst.append( Thetabl )

		# (Theta,b), going to output layer, l=L
		if (L>2):
			inputlayer_al = Thetabs_lst[-1].alp1

			#initialize an instance of class Thetab_right
			Thetabl = Thetab_right(L-1,(s_l[L-2],s_l[L-1]),al=inputlayer_al, activation=psi_Lm1, rng=rng)
			Thetabl.connect_through()
			Thetabs_lst.append(Thetabl)
		
		self.Thetabs = Thetabs_lst
		
		# note that in the context of LSTM, h_Theta here is g_t
		h_Theta = self.Thetabs[-1].alp1 # output layer, which should be given the values for g_t for LSTM

	def connect_through(self, X_t, h_tm1):
		""" connect_through - connect through the layers (the actual feedforward operation)

		INPUTS/PARAMETERS
		=================
		@type X_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		@type y_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		"""
		self.Thetabs[0].al = X_t
		self.Thetabs[0]._h  = h_tm1
		self.Thetabs[0].connect_through()
		
		L = self.L
		
		for l in range(2,L):  	# l=2,3,...L-1, for each of the Theta operations between layers l
			self.Thetabs[l-1].al = self.Thetabs[l-2].alp1
			self.Thetabs[l-1].connect_through()	

		
		# return the result of Feedforward operation, denoted h_Theta
		h_Theta = self.Thetabs[-1].alp1
		return h_Theta


""" FeedForward_ifo_right - this is essentially the same as FeedForward_g_right, but with these changes:
	* the "starting" "type" of layer is Theta,b,theta,W, as opposed to Theta,b,theta, "only".  
		So the input arguments will be different (1 more)				
"""
class Feedforward_ifo_right(object):
	""" Feedforward - Feedforward
	"""
	def __init__(self, L, s_l, H, activation_fxn=T.tanh, psi_Lm1=T.tanh, rng=None ):
		""" Initialize MLP class

		INPUT/PARAMETER(S)
		==================
		@type L : (positive) integer
		@param L : total number of layers L, counting l=1 (input layer), and l=L (output layer), so only 1 hidden layer means L=3

		@type s_l : Python list (or tuple, or iterable) of (positive) integers
		@param s_l : list (or tuple or iterable) of length L, containing (positive) integers for s_l, size or number of "nodes" or "units" in layer l=1,2,...L; because of how Python counts from 0, it's indexed 0,1,...L-1
					NOTE that number of "components" of y, K, must be equal to s_L, s_L=K
						
		@type H    : (positive) integer, 
		@param H   : number of components of hidden data				
		"""
		self.L = L 
		self.s_l = s_l
	
		if rng is None:
			rng = np.random.RandomState(1234)
	
		###############
		# BUILD MODEL #
		###############
		Thetabs_lst = [] # temporary list of Thetas, Theta,b weights or parameters

		# initialize an instance of class ThetabthetaW_right
		ThetabthetaW1 = ThetabthetaW_right( (s_l[0],s_l[1]),H, activation=activation_fxn, l=1, rng=rng)
		ThetabthetaW1.connect_through()
		
		Thetabs_lst.append( ThetabthetaW1 )
		
		for l in range(2,L-1): # don't include the Theta,b going to the output layer in this loop
			inputlayer_al = Thetabs_lst[-1].alp1
		
			# initialize an instance of class Thetab_right
			Thetabl = Thetab_right(l,(s_l[l-1],s_l[l]),al=inputlayer_al, activation=activation_fxn, rng=rng)
			Thetabl.connect_through()
			Thetabs_lst.append( Thetabl )
		
		# (Theta,b), going to output layer, l=L
		if ( L>2):
			inputlayer_al = Thetabs_lst[-1].alp1

			# initialize an instance of class Thetab_right
			Thetabl = Thetab_right(L-1,(s_l[L-2],s_l[L-1]),al=inputlayer_al, activation=psi_Lm1, rng=rng)
			Thetabl.connect_through()
			Thetabs_lst.append(Thetabl)
		
		self.Thetabs = Thetabs_lst
		
		# note that in the context of LSTM, h_Theta here is g_t,i_t,f_t
		h_Theta = self.Thetabs[-1].alp1 # output layer, which should be given the values for g_t,i_t,f_t for LSTM

	def connect_through(self, X_t, h_tm1, c_tm1):
		""" connect_through - connect through the layers (the actual feedforward operation)

		INPUTS/PARAMETERS
		=================
		@type X_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		@type y_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		"""
		self.Thetabs[0].al = X_t
		self.Thetabs[0]._h = h_tm1
		self.Thetabs[0]._c = c_tm1
		self.Thetabs[0].connect_through()

		L = self.L
		
		for l in range(2,L): # l=2,3,...L-1, for each of the Theta operations between layers l
			self.Thetabs[l-1].al = self.Thetabs[l-2].alp1
			self.Thetabs[l-1].connect_through()
			
		# return the result of the gate (\alpha), denoted h_Theta	
		h_Theta = self.Thetabs[-1].alp1
		return h_Theta


class LSTM_Model_right(object):
	""" LSTM_Model - represents the LSTM (Long Short Term Memory) model, with its g,i,f, and o "gates"

	(AVAILABLE) CLASS MEMBERS
	=========================
	@type .L   : namedtuple of (positive) integers, with entries 'g','i','f','o', in that order
	@param .L : number of layers each for gates g, i, f, o
	
	@type s_l_in : namedtuple of Python lists (or tuple, or iterable) of (positive) integers
	@param s_l_in : size or number of "nodes" or units for all the layers L^{(\alpha)} for each of the (4) gates, 'g','i','f','o', 
		e.g. \alpha = { 'g','i','o','f' }, let l^{(\alpha})=1,2,...L^{(\alpha)}, then s_l^{\alpha} has the number of nodes or units for that layer l, for that "gate" \alpha

	@type .psi : namedtuple of tuples, each tuple (of length 2) of theano functions (or numerical numpy functions), with entries 'g','i','f','o','h'
	@param .psi : sigmoidal functions necessary for LSTM, denoted psi's


	@type .W  : theano shared of size dim. K, K, i.e. \theta \in \text{Mat}_{\mathbb{R}}(K,K)
	@param .W : weight, that multiplies with "cell memory" value from t-1 

	@type ._gates : Python named tuple of Python class instances
	@param ._gates : named tuple each contains a Feedforward (for .g) or Feedforward_ifo (for .i,.f,.o) class instance that 
					represents a "gate"

	@type .params : list of Theano shared variables,
	@params .params : list of Theano shared variables representing all the parameters that are variables in our model
	
	@type .Thetas_only : list of Theano shared variables
	@params .Thetas_only : list of Theano shared variables of only the Theta,theta,W weights, 
							because they're the only terms used for regularizations 
							(you don't want to include the intercepts because the don't multiply the input data X)

	NOTES
	=====
	
		
	"""

	def __init__(self, L_in, s_l_in, H,K,activations, activation_y, rng=None):
		""" Initialize LSTM class
		
		INPUT/PARAMETER(S)
		==================  

		@type L_in : namedtuple of (positive) integers, with entries 'g','i','f','o', in that order
		@param L_in : number of layers each for gates g, i, f, o
		
		@type s_l_in : namedtuple of Python lists (or tuple, or iterable) of (positive) integers
		@param s_l_in : size or number of "nodes" or units for all the layers L^{(\alpha)} for each of the (4) gates, 'g','i','f','o', 
			e.g. \alpha = { 'g','i','o','f' }, let l^{(\alpha})=1,2,...L^{(\alpha)}, then s_l^{\alpha} has the number of nodes or units for that layer l, for that "gate" \alpha

		@type H : (positive) integers
		@param H : number of componenets of "hidden layer units" h_t, h_t \in \mathbb{R}^H

		@type K : (positive) integers
		@param K : number of componenets of output y, y \in \mathbb{R}^K
		
		@type activations : namedtuple of tuples, each tuple (of length 2) of theano functions (or numerical numpy functions), with entries 'g','i','f','o','h'
		@param activations : sigmoidal functions necessary for LSTM, denoted psi's
		"""
		
		self.L = L_in
		self.s_l = s_l_in

		if rng is None:
			rng = np.random.RandomState(1234)

		self.psis = activations
		
		###############
		# BUILD MODEL #
		###############
#Psis = namedtuple("Psis",['g','i','f','o','h'])

		
		# ._gates contains the 'g','i','f','o' gates, input data,input,forget, and output gates, respectively
		self._gates = Gates( 
			g= Feedforward_g_right( L_in.g, s_l_in.g, H, activation_fxn= activations.g[0], 
								psi_Lm1=activations.g[-1], rng=rng), 
			i= Feedforward_ifo_right( L_in.i, s_l_in.i, H, activation_fxn= activations.i[0], 
								psi_Lm1=activations.i[-1], rng=rng), 
			f= Feedforward_ifo_right( L_in.f, s_l_in.f, H, activation_fxn= activations.f[0], 
								psi_Lm1=activations.f[-1], rng=rng), 
			o= Feedforward_ifo_right( L_in.o, s_l_in.o, H, activation_fxn= activations.o[0], 
								psi_Lm1=activations.o[-1], rng=rng)  
		)

		s_L = s_l_in.g[-1]  # number of "units" or "nodes" in layer L^{(g)} of gate g, the last layer before going to h

		# instantiate Thetab_right class to represent h_t -> y_t
		self.Thetaby = Thetab_right( 1, (s_L,K), activation=activation_y, rng=rng)



		## unroll all the parameters
		
		gThetas = [Weight.Theta for Weight in self._gates.g.Thetabs]
		gbs     = [Weight.b for Weight in self._gates.g.Thetabs]
		gthetas = [Weight.theta for Weight in self._gates.g.Thetabs]

#		gthetas = []
#		for Weight in self._gates.g.Thetabs:
#			try:
#				gthetas.append( Weight.theta )
#			except AttributeError:
#				print("on layer l=%d" % Weight.l)
				
		self.params = gThetas + gbs + gthetas
		
		self.Thetas_only = gThetas + gthetas
		
		iThetas = [Weight.Theta for Weight in self._gates.i.Thetabs]
		ibs     = [Weight.b for Weight in self._gates.i.Thetabs]
		ithetas = [Weight.theta for Weight in self._gates.i.Thetabs]
		iWs     = [Weight.W for Weight in self._gates.i.Thetabs]

#		ithetas=[]
#		for Weight in self._gates.i.Thetabs:
#			try:
#				ithetas.append( Weight.theta )
#			except AttributeError:
#				print("on layer l=%d" % Weight.l)

#		iWs=[]
#		for Weight in self._gates.i.Thetabs:
#			try:
#				iWs.append( Weight.theta )
#			except AttributeError:
#				print("on layer l=%d" % Weight.l)


		self.params = self.params + iThetas + ibs + ithetas + iWs

		self.Thetas_only = self.Thetas_only + iThetas + ithetas + iWs

		
		fThetas = [Weight.Theta for Weight in self._gates.f.Thetabs]
		fbs     = [Weight.b for Weight in self._gates.f.Thetabs]
		fthetas = [Weight.theta for Weight in self._gates.f.Thetabs]
		fWs     = [Weight.W for Weight in self._gates.f.Thetabs]
#		fthetas=[]
#		for Weight in self._gates.f.Thetabs:
#			try:
#				fthetas.append( Weight.theta )
#			except AttributeError:
#				print("on layer l=%d" % Weight.l)

#		fWs=[]
#		for Weight in self._gates.f.Thetabs:
#			try:
#				fWs.append( Weight.theta )
#			except AttributeError:
#				print("on layer l=%d" % Weight.l)


		self.params = self.params + fThetas + fbs + fthetas + fWs

		self.Thetas_only = self.Thetas_only + fThetas + fthetas + fWs

		
		oThetas = [Weight.Theta for Weight in self._gates.o.Thetabs]
		obs     = [Weight.b for Weight in self._gates.o.Thetabs]
		othetas = [Weight.theta for Weight in self._gates.o.Thetabs]
		oWs     = [Weight.W for Weight in self._gates.o.Thetabs]
#		othetas=[]
#		for Weight in self._gates.o.Thetabs:
#			try:
#				othetas.append( Weight.theta )
#			except AttributeError:
#				print("on layer l=%d" % Weight.l)

#		oWs=[]
#		for Weight in self._gates.o.Thetabs:
#			try:
#				oWs.append( Weight.theta )
#			except AttributeError:
#				print("on layer l=%d" % Weight.l)


		self.params = self.params + oThetas + obs + othetas + oWs

		self.Thetas_only = self.Thetas_only + oThetas + othetas + oWs


		self.params = self.params + [ self.Thetaby.Theta, self.Thetaby.b ]

		self.Thetas_only = self.Thetas_only + [self.Thetaby.Theta, ]

		print "Total number of parameters: %d " % len(self.params) 
		
		
		
	def build_lstm_step(self):
		""" build_lstm_step - returns a function, named lstm_step, that executes (1) LSTM step
		
		lstm_step = lstm_step(x_t, h_tm1, c_tm1)
		"""

		def lstm_step(X_t, h_tm1, c_tm1, *args_for_params): 

			g_t = self._gates.g.connect_through(X_t, h_tm1)
			i_t = self._gates.i.connect_through(X_t, h_tm1, c_tm1)
			f_t = self._gates.f.connect_through(X_t, h_tm1, c_tm1)
			
			c_t = f_t * c_tm1 + i_t * g_t

			o_t = self._gates.o.connect_through(X_t, h_tm1, c_t)
			h_t = o_t * self.psis.h[-1]( c_t )
			
			self.Thetaby.al = h_t
			self.Thetaby.connect_through()
			
			y_t = self.Thetaby.alp1
			
			return [h_t, c_t, y_t]
	
		return lstm_step


class MemoryBlock_right(object):
	""" MemoryBlock_right
	
	(AVAILABLE) CLASS MEMBERS
	=========================
	
	"""
	def __init__(self, H, LSTM_model, X=None, y=None,h0=None,c0=None):
		""" Initialize MemoryBlock class
		
		INPUT/PARAMETER(S)
		==================  
		@type H : (positive) integers
		@param H : number of componenets of "hidden layer units" h_t, h_t \in \mathbb{R}^H

		@type LSTM_model : LSTM_Model class instance
		@param LSTM_model : LSTM_Model class instance
		
		"""
		self.LSTM_model = LSTM_model

		if X is None:
			X = T.matrix(dtype=theano.config.floatX)

		self.X = X

		if y is None:
			y = T.matrix(dtype=theano.config.floatX)

		self.y = y

		self._c0 = theano.shared(np.zeros(H).astype(theano.config.floatX) )
		self._h0 = T.tanh( self._c0)
		
		
	def build_scan_over_t(self):
		""" build_scan_over_t
		"""
		lstm_step_fxn = self.LSTM_model.build_lstm_step()
		X = self.X
		c0 = self._c0
		h0 = self._h0
		params= self.LSTM_model.params
	
		##############################
		# Programming note on GPU
		##############################
	
		""" - allow_gc If you use cnmem and this scan is on GPU, the speed up from the scan allow_gc is small. 
		If you are missing memory, disable the scan allow_gc could help you run graph that request much memory. """
		""" 
		http://deeplearning.net/software/theano/library/config.html
		This sets the default for the use of the Theano garbage collector for intermediate results. 
		To use less memory, Theano frees the intermediate results as soon as they are no longer needed. 
		Disabling Theano garbage collection allows Theano to reuse buffers for intermediate results between function calls. 
		This speeds up Theano by no longer spending time reallocating space. 
		This gives significant speed up on functions with many ops that are fast to execute, 
		but this increases Theano's memory usage.
		For gpu, may have to set allow gc to be False
		or even
		assert_no_cpu_op='raise'
		in 
		THEANO_FLAGS="float32,device=gpu,assert_no_cpu_op='raise'" python test.py
		"""
			
		[h_vals, c_vals, y_vals], updates_from_scan = theano.scan(fn=lstm_step_fxn, 
					sequences=dict(input=X, taps=[0]),
					outputs_info = [h0,c0,None],
					non_sequences = params)
	
	
		self.scan_res = [h_vals, c_vals, y_vals], updates_from_scan
		return [h_vals, c_vals, y_vals], updates_from_scan

	def build_J(self, lambda_val):
		""" build_J - build or make cost functional
		"""
		y_sym = self.y
		
		lambda_val = np.float32( lambda_val )  # regularization constant
		J = build_cost_functional( lambda_val, 
									self.scan_res[0][-1], # we want y_vals from above, predicted value for y
									y_sym, self.LSTM_model.Thetas_only)
		self.J_Theta = J
		return J

	def build_update(self,alpha=0.01,beta=0.0):
		J = self.J_Theta
		Thetabs = self.LSTM_model.params
		X = self.X
		y = self.y
		
		self.updateExpression, self.gradDescent_step = build_gradDescent_step(J, Thetabs,X,y,alpha,beta)

	
	###############
	# TRAIN MODEL #
	###############	
		
	def train_model_full(self, train_data, max_iters=250):
		print "theano.config.allow_gc =: " , theano.config.allow_gc

		train_errors = np.ndarray(max_iters)

		learn_rnn = self.gradDescent_step
		
		m = len(train_data)
		
		for iter in range( max_iters) :
			error = 0.
			for j in range(m):
				X,y = train_data[j]
#				X=X.astype(theano.config.floatX)
#				y=y.astype(theano.config.floatX)
				J_train = learn_rnn(X,y) # cost of training, train_cost
#				error += J_train
#			train_errors[iter] = error
		return train_errors

	""" Dr. Herta's original implementation - uses random to select which time t to train on
	"""
	def train_rnn(self, train_data, max_iters=250, verbose=False):
		print "theano.config.allow_gc =: " , theano.config.allow_gc
		train_errors = np.ndarray(max_iters)

		learn_rnn_fn = self.gradDescent_step
		
		m = len(train_data)
		
		for iter in range(max_iters):
#			print "This is t=%d \n" % t 
			
			if verbose: # sanity check
				print "t=%d " % iter

			error = 0.
			
			if verbose: # sanity check
				print " j="

			for j in range(m):
#				print "\t This is j=%d \n" % j
				
				if verbose: # sanity check
					print j
				
				index = np.random.randint(0, m)
				i, o = train_data[index]
				train_cost = learn_rnn_fn(i, o)
				error += train_cost
			train_errors[iter] = error
		
		return train_errors
	

	####################
	# Predicted values #
	####################
	def prediction_fxn(self):
		X = self.X
		y_predicted = self.scan_res[0][-1] # we want y_vals from above, from build_scan_over_t, predicted value for y

		predictions = theano.function(inputs=[X], outputs=y_predicted)
		return predictions
		
#	def predict(self,X_vals):
#		prediction_func = self.prediction_fxn()
#		y_predicted = prediction_func(X_vals)
#		return y_predicted

	def predict_on_lst(self, test_data, verbose=True):
		"""
		@type test_data : Python list (of length m, number of training examples)

		@type verbose : bool
		@param verbose : boolean flag saying to do the print outs (be "verbose"; True) or not (False)
		"""

		predictions = []
		
		for i,o in test_data:  # test_data is a list from j=0,1,...m-1, m total training data points
			predictions_func = self.prediction_fxn()
			predicted_y = predictions_func(i)  # i is a list of arrays of size d features, list of length T, for time t=0,1...T-1

		# EY : 20170223 Try this in the future; uncomment it out
#			i = np.array(i).astype(theano.config.floatX)
#			predicted_y = predictions_func(i)
			
			if verbose:
				print o[-2] # target
				print predicted_y[-2] # prediction
			
			predictions.append( predicted_y )
			
		return predictions
	
	
	#################################
	## for Long-Term Serialization ##
	## cf. http://deeplearning.net/software/theano/tutorial/loading_and_saving.html#long-term-serialization
	#################################
	
	def __getstate__(self):
		""" __getstate__(self)  
		
		returns a Python list of numpy arrays, (through theano.shared.get_value() )
		"""
		
		params = self.LSTM_model.params
		params_vals = [weight.get_value() for weight in params]
		return params_vals
		
	def __setstate__(self, weights):
		""" __setstate__(self,weights)
		
		@type weights : list of numpy arrays (length must correspond exactly with LSTM model)
		@param weights : list of numpy arrays each representing values for the (trained) weights
		
		"""
		number_of_weights = len(weights)

		# EY : 20170223 note, figure how to enforce this sanity check
		# assert number_of_weights == len( self.LSTM_model.params )

		for ith in range(number_of_weights):
			self.LSTM_model.params[ith].set_value( weights[ith] )
			
	def save_parameters(self, filename='objects.save'):
		""" save_parameters (save the weights or parameters of this model as numpy arrays that are pickled)
		
		@type filename : string
		@param filename : string to name the saved file
		"""	
		f = open(filename,'wb')
		for param in self.LSTM_model.params:
			cPickle.dump( param.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()



def build_cost_functional(lambda_val, h, y_sym, Thetas):
	""" build_cost_functional (with regularization) J=J_y(Theta,b) # J\equiv J_y(\Theta,b), but now with 
	X,y being represented as theano symbolic variables first, before the actual numerical data values are given

	INPUT/PARAMETERS
	================
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

		# logistic regression cost function J, with no regularization (yet)
#	J_theta = T.mean( T.sum(
#			- y_sym * T.log(h) - (np.float32(1)-y_sym) * T.log( np.float32(1) - h), axis=0), axis=0)
	J_theta = - T.mean( y_sym * T.log(h) + (np.float32(1.) - y_sym)*T.log( np.float32(1.) - h) )

	reg_term = np.float32(lambda_val/ (2. )) /m *T.sum( [ T.sum( Theta*Theta) for Theta in Thetas] )

	J_theta = J_theta + reg_term 
	return J_theta
	

def build_gradDescent_step( J, Thetabs, X_sym,y_sym, alpha =0.01, beta = 0.0):
	""" build_gradDescent_step - gradient Descent (with momentum), but from build_cost_functional for the J

	INPUT/PARAMETERS:
	=================
	@param J     : cost function (from build_cost_function)
	
	@type Thetabs  : Python list (or tuple or iterable) of Thetas, weights matrices, and intercept "row" arrays or theano equivalent
	@param Thetabs : weights or i.e. parameters
	
	@type X_sym   : theano symbolic variable, such as T.matrix()
	@param X_sym  : theano symbolic variable representing input X data 
	
	@type y_sym   : theano symbolic variable, such as T.matrix()
	@param y_sym  : theano symbolic variable representing output y data, outcomes
	
	
	@param alpha : learning rate
	
	@param beta  : "momentum" constant (parameter)

	RETURN(S)
	=========
	@type updateThetas, gradientDescent_step : tuple of (list of theano symbolic expression, theano function)

	"""
	# this is ok
	updateThetabs = [ sandbox.cuda.basic_ops.gpu_from_host( 
					Theta - np.float32( alpha) * T.grad( J, Theta) + np.float32(beta)*Theta ) for Theta in Thetabs]
	

#	gradientDescent_step = theano.function([],
#											updates=zip(Thetabs,updateThetabs),
#											givens={ X_sym : X_vals.astype(theano.config.floatX), 
#													y_sym : y_vals.astype(theano.config.floatX) },
#											name="gradDescent_step")
	gradientDescent_step = theano.function(inputs = [X_sym, y_sym],
											outputs = J, 
											updates = zip(Thetabs,updateThetabs) )

	return updateThetabs, gradientDescent_step
		
	




	
########################################
##########  __main__  		  ##########
########################################

if __name__ == "__main__":
	print("In main.")

	L_Herta = Gates(g=2,i=2,f=2,o=2)
	
	n_hidden = n_i = n_c = n_o = n_f = 10
	n_in = 7 # for embedded reber grammar
	n_y = 7 # for embedded reber grammar; this is K in my notation
	
	s_l_Herta = Gates(g=[n_in,n_c],i=[n_in,n_i],f=[n_in,n_f],o=[n_in,n_o])
	
	activations_Herta = Psis(g=(T.tanh, T.tanh), 
							i=(T.nnet.sigmoid, T.nnet.sigmoid),
							f=(T.nnet.sigmoid, T.nnet.sigmoid),
							o=(T.nnet.sigmoid, T.nnet.sigmoid),
							h=(T.tanh,)
							)

	# sigma_y = T.nnet.sigmoid in this case
	
	LSTM_model_Herta= LSTM_Model_right( L_Herta, s_l_Herta, n_hidden, n_y, activations_Herta, T.nnet.sigmoid)
	lstm_step_fxn = LSTM_model_Herta.build_lstm_step()
	MemBlck_Herta = MemoryBlock_right(n_hidden, LSTM_model_Herta)
	MemBlck_Herta.build_scan_over_t()
	MemBlck_Herta.build_J(0.1)
	
	# this may take a while below, so I commented it out; uncomment to actually do it
	#MemBlck_Herta.build_update()

	import sys
	sys.path.append('../')
	try:
		import reberGrammar
		train_data = reberGrammar.get_n_embedded_examples(1000)
	except ImportError:
		print "Import error, and so we need to locate the file path, manually \n"


