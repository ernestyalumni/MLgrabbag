"""
@file LSTM.py
@brief LSTM - Long Short-Term Memory with multiple layers

@author Ernest Yeung
@email ernestyalumni dot gmail dot com
"""  

import theano
import numpy as np
import theano.tensor as T
from theano import sandbox

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

		if rng is None:
			rng = np.random.RandomState(1234)


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
			al = T.matrix()
			
			
		self.Theta = Theta  # size dims. (s_l,s_lp1) i.e. s_lp1 x s_l
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
		lin_zlp1 = T.dot( self.Theta, self.al)+T.tile(self.b, (1,self.al.shape[1].astype('int32') ) ) # z^{(l+1)}
		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )


class Thetabtheta(object):
	""" Thetabtheta - Thetabtheta class for parameters or weights and intercepts b between layers 1 and 2 for LSTM

	(AVAILABLE) CLASS MEMBERS
	=========================
	@type .Theta  : theano shared; of size dims. (s_1,s_{2}), i.e. matrix dims. (s_{2}x(s_1)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_1,s_{2}) 
	@param .Theta : "weights" or parameters for l=1
		
	@type .b 	  : theano shared of dim. s_{2} 
	@param .b     : intercepts

	@type .theta  : theano shared of size dim. s_{2}, K, i.e. \theta \in \text{Mat}_{\mathbb{R}}(K,s_{2})
	@param .theta : weight 

	@type .al 	  : theano shared variable or vector of size dims. (s_l, m), m=1,2..., number of training examples
	@param .al    : "nodes" or "units" of layer l
	
	@type ._y_sym  : theano matrix
	@param ._y_sym : symbolic variable representing output data of training examples
	
	@type .alp1   : theano symbolic expression for array of size dims. (s_lp1,m)
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

	def __init__(self, s_ls, y=None, al=None, Theta=None, b=None, theta=None, activation=T.tanh, K=1, l=1, rng=None):
		""" Initialize the parameters for the `layer`
		
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

		@type K    : (positive) integer, default is 1
		@param K   : number of components of output data

		@type l    : (positive) integer, default is 1
		@param l   : layer number label, l=1 (input),2,...L-1, L is "output layer"

		@type rng  : numpy.random.RandomState
		@param rng : random number generator used to initialize weights

		"""
		s_2, s_1 = s_ls

		if rng is None:
			rng = np.random.RandomState(1234)


		if Theta is None:
			Theta_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( s_1 + s_2 )), 
					high=np.sqrt(6. / ( s_1 + s_2 )), size=(s_2, s_1) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				Theta_values *= np.float32( 4 )
			
			Theta = theano.shared(value=Theta_values, name="Theta"+str(l), borrow=True)

		if theta is None:
			theta_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( K + s_2 )), 
					high=np.sqrt(6. / ( K + s_2 )), size=(s_2, K) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				theta_values *= np.float32( 4 )
			
			theta = theano.shared(value=theta_values, name="theta"+str(l), borrow=True)



		if b is None:
			b_values = np.vstack( np.zeros(s_2)).astype(theano.config.floatX)
			b= theano.shared(value=b_values, name='b'+str(l), borrow=True)	
			
		if al is None:
			al = T.matrix()
		
		if y is None:
			y = T.matrix()	
			
		self.Theta = Theta  # size dims. (s_1,s_2) i.e. s_2 x s_1
		self.b     = b      # dims. s_2
		self.theta = theta  # size dims. s_2 x K i.e. theta \in \text{Mat}_{\mathbb{R}}( K,s_2)
		self.al    = al     # dims. s_1
		self.y     = y      # dims. K
		
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
		lin_zlp1 = T.dot( self.Theta, 
							self.al)+T.tile(self.b, 
											(1,self.al.shape[1].astype('int32'))) + T.dot( self.theta,
																							self.y) # z^{(l+1)}
		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )


class Feedforward(object):
	""" Feedforward - Feedforward
	"""
	def __init__(self, L, s_l, activation_fxn=T.tanh, psi_Lm1=T.tanh, rng=None ):
		""" Initialize MLP class

		INPUT/PARAMETER(S)
		==================
		@type L : (positive) integer
		@param L : total number of layers L, counting l=1 (input layer), and l=L (output layer), so only 1 hidden layer means L=3

		@type s_l : Python list (or tuple, or iterable) of (positive) integers
		@param s_l : list (or tuple or iterable) of length L, containing (positive) integers for s_l, size or number of "nodes" or "units" in layer l=1,2,...L; because of how Python counts from 0, it's indexed 0,1,...L-1
					NOTE that number of "components" of y, K, must be equal to s_L, s_L=K
						
		
		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		"""
		self.L = L 
		self.s_l = s_l
	
		if rng is None:
			rng = np.random.RandomState(1234)
	
		###############
		# BUILD MODEL #
		###############

		Thetabs_lst = [] # temporary list of Thetas, Theta,b weights or parameters
		
		K_in = self.s_l[-1]
		Thetabtheta1 = Thetabtheta( (s_l[1],s_l[0]),activation=activation_fxn, K= K_in, l=1, rng=rng)
		Thetabtheta1.connect_through()
		
		Thetabs_lst.append( Thetabtheta1 )
		
		for l in range(2,L-1): # don't include the Theta,b going to the output layer in this loop
			inputlayer_al = Thetabs_lst[-1].alp1
			Thetabl = Thetab(l,(s_l[l],s_l[l-1]),al=inputlayer_al, activation=activation_fxn, rng=rng)
			Thetabl.connect_through()
			Thetabs_lst.append( Thetabl )
		
		# (Theta,b), going to output layer, l=L
		if (L>2):
			inputlayer_al = Thetabs_lst[-1].alp1
			Thetabl = Thetab(L-1,(s_l[L-1],s_l[L-2]),al=inputlayer_al, activation=psi_Lm1, rng=rng)
			Thetabl.connect_through()
			Thetabs_lst.append(Thetabl)
		
		self.Thetabs = Thetabs_lst
		
		# note that in the context of LSTM, h_Theta here is g_t,i_t,f_t
		h_Theta = self.Thetabs[-1].alp1 # output layer, which should be given the values for g_t,i_t,f_t for LSTM

	def connect_through(self, X_t, h_tm1):
		""" connect_through - connect through the layers (the actual feedforward operation)

		INPUTS/PARAMETERS
		=================
		@type X_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		@type y_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		"""
		self.Thetabs[0].al = X_t
		self.Thetabs[0].y  = h_tm1
		self.Thetabs[0].connect_through()
		
		for l in range(2,L): # l=2,3,...L-1, for each of the Theta operations between layers l
			self.Thetabs[l-1].al = self.Thetabs[l-1].alp1
			self.Thetabs[l-1].connect_through()	
		
		# return the result of Feedforward operation, denoted h_Theta
		h_Theta = self.Thetabs[-1].alp1
		return h_Theta
		

class LSTM_t(object):
	""" LSTM_t - class for LSTM, Long Short-Term Memory, at time t
	"""
	def __init__(self, L_in, s_l_in, activations,X_sym=None, y_sym=None, lambda_val_in=1.,rng_in=None):
		""" Initialize LSTM class
		
		INPUT/PARAMETER(S)
		==================  

		@type L_in : namedtuple of (positive) integers, with entries 'g','i','f','o', in that order
		@param L_in : number of layers each for gates g, i, f, o
		
		@type s_l_in : namedtuple of Python lists (or tuple, or iterable) of (positive) integers
		@param s_l_in : size or number of "nodes" or units for all the layers L^{(\alpha)} for each of the (4) gates, 'g','i','f','o', 
			e.g. \alpha = { 'g','i','o','f' }, let l^{(\alpha})=1,2,...L^{(\alpha)}, then s_l^{\alpha} has the number of nodes or units for that layer l, for that "gate" \alpha
		
		@type activations : namedtuple of theano functions (or numerical numpy functions), with entries 'g','i','f','o','h'
		@param activations : sigmoidal functions necessary for LSTM, denoted psi's
		"""
		
		self.L = L_in
		self.s_l = s_l_in

		# symbolic Theano variable
		if X_sym is None:
			self._X_sym = T.matrix() # X input data represented as a symbolic variable
		else:
			self._X_sym = X_sym
		if y_sym is None:
			self._y_sym = T.matrix() # y output data represented as a symbolic variable 
		else:
			self._y_sym = y_sym
		
		
		if rng_in is None:
			rng = np.random.RandomState(1234)
		else:
			rng= rng_in
		
		###############
		# BUILD MODEL #
		###############
		
#Psis = namedtuple("Psis",['g','i','f','o','h'])
		
		

		# ._gates contains the 'g','i','f','o' gates, input data,input,forget, and output gates, respectively
		self._gates = Gates( 
			g= Feedforward( L_in.g, s_l_in.g, X_sym=self._X_sym, y_sym=self._y_sym, 
							activation_fxn= activations.g[0], psi_Lm1=activations.g[-1], 
							lambda_val=lambda_val_in, rng_in=rng), 
			i= Feedforward( L_in.i, s_l_in.i, X_sym=self._X_sym, y_sym=self._y_sym, 
							activation_fxn= activations.i[0], psi_Lm1=activations.i[-1], 
							lambda_val=lambda_val_in, rng_in=rng), 
			f= Feedforward( L_in.f, s_l_in.f, X_sym=self._X_sym, y_sym=self._y_sym, 
							activation_fxn= activations.f[0], psi_Lm1=activations.f[-1], 
							lambda_val=lambda_val_in, rng_in=rng), 
			o= Feedforward( L_in.o, s_l_in.o, X_sym=self._X_sym, y_sym=self._y_sym, 
							activation_fxn= activations.o[0], psi_Lm1=activations.o[-1], 
							lambda_val=lambda_val_in, rng_in=rng)  
		)

		
##########################################################################################
## Dr. Christian Herta's LSTM
##########################################################################################
""" ThetabthetaW - for gates i,f,o, the input, forget, and output gates, respectively
"""
class ThetabthetaW(object):
	""" Thetabtheta - Thetabtheta class for parameters or weights and intercepts b between layers 1 and 2 for LSTM

	(AVAILABLE) CLASS MEMBERS
	=========================
	@type .Theta  : theano shared; of size dims. (s_1,s_{2}), i.e. matrix dims. (s_{2}x(s_1)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_1,s_{2}) 
	@param .Theta : "weights" or parameters for l=1
		
	@type .b 	  : theano shared of dim. s_{2} 
	@param .b     : intercepts

	@type .theta  : theano shared of size dim. s_{2}, K, i.e. \theta \in \text{Mat}_{\mathbb{R}}(K,s_{2})
	@param .theta : weight 

	@type .W  : theano shared of size dim. K, K, i.e. \theta \in \text{Mat}_{\mathbb{R}}(K,K)
	@param .W : weight, that multiplies with "cell memory" value from t-1 


	@type .al 	  : theano shared variable or vector of size dims. (s_l, m), m=1,2..., number of training examples
	@param .al    : "nodes" or "units" of layer l
		
	@type .alp1   : theano symbolic expression for array of size dims. (s_lp1,m)
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

	def __init__(self, s_ls, H, h_tm1=None, al=None, c_tm1=None, Theta=None, b=None, theta=None, W=None,
					activation=T.tanh, l=1, rng=None):
		""" Initialize the parameters for the `layer`
		
		@type s_ls : tuple of (positive) integers of size (length) 2, only
		@param s_ls : "matrix" size dimensions of Theta or weight mmatrix, of size dims. (s_lp1, s_l) (this is important)
		Bottom line: s_ls = (s_lp1, s_l)

		@type H    : (positive) integer, default is 1
		@param H   : number of components of "hidden" cell memory data

		@type al     : theano shared variable or vector of size dims. (s_l, m), m=1,2..., number of training examples
		@param al    : "nodes" or "units" of layer l

		@param c_tm1 : c_tm1, c_{t=1} for "cell memory" at t-1 time

		@type Theta  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l+1}x(s_l)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param Theta : "weights" or parameters for l, l=1,2, ... L-1
		
		@type b      : theano shared of dim. s_{l+1} or s_lp1
		@param b     : intercepts

		@type theta  : theano shared; of size dims. (s_{l+1},K), i.e. matrix dims. (s_{l+1}x(K)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(K,s_{l+1}) 
		@param theta : "weights" or parameters, multiplying y_{t-1}, for l, l=1,2, ... L-1


		@type activation  : theano.Op or function
		@param activation : Non linearity to be applied in the layer(s)


		@type l    : (positive) integer, default is 1
		@param l   : layer number label, l=1 (input),2,...L-1, L is "output layer"

		@type rng  : numpy.random.RandomState
		@param rng : random number generator used to initialize weights

		"""
		s_lp1, s_l = s_ls

		if rng is None:
			rng = np.random.RandomState(1234)


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

		if theta is None:
			theta_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( H + s_lp1 )), 
					high=np.sqrt(6. / ( H + s_lp1 )), size=(s_lp1, H) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				theta_values *= np.float32( 4 )
			
			theta = theano.shared(value=theta_values, name="theta"+str(l), borrow=True)



		if b is None:
			b_values = np.vstack( np.zeros(s_lp1)).astype(theano.config.floatX)
			b= theano.shared(value=b_values, name='b'+str(l), borrow=True)	

		if W is None:
			W_values = np.asarray( 
				rng.uniform( 
					low=-np.sqrt(6. / ( H + s_lp1 )), 
					high=np.sqrt(6. / ( H + s_lp1 )), size=(s_lp1, H) ), 
					dtype=theano.config.floatX 
			)
			if activation == T.nnet.sigmoid:
				W_values *= np.float32( 4 )
			
			W = theano.shared(value=W_values, name="W"+str(l), borrow=True)

			
		if al is None:
			al = T.matrix().astype(theano.config.floatX)
		
		if h_tm1 is None:
			h_tm1 = T.matrix().astype(theano.config.floatX)	
		
		if c_tm1 is None:
			c_tm1 = T.matrix().astype(theano.config.floatX)
			
		self.Theta = Theta  # size dims. (s_1,s_2) i.e. s_2 x s_1
		self.b     = b      # dims. s_2
		self.theta = theta  # size dims. s_2 x H i.e. theta \in \text{Mat}_{\mathbb{R}}( H,s_2)
		self.W     = W		# size dims. s_2 x H
		self.al    = al     # dims. s_1
		self.h_tm1     = h_tm1      # dims. H
		
		self.c_tm1 = c_tm1
		
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
		lin_zlp1 = T.dot( self.Theta, 
						self.al)+T.tile(self.b, 
									(1,
									self.al.shape[1].astype('int32'))) + T.dot( self.theta,
																			self.h_tm1) + T.dot( self.W,
																T.tile( self.c_tm1, (1,self.al.shape[1].astype('int32'))))  
																			# z^{(l+1)}
		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )


class Feedforward_ifo(object):
	""" Feedforward - Feedforward
	"""
	def __init__(self, L, s_l, activation_fxn=T.tanh, psi_Lm1=T.tanh, rng=None ):
		""" Initialize MLP class

		INPUT/PARAMETER(S)
		==================
		@type L : (positive) integer
		@param L : total number of layers L, counting l=1 (input layer), and l=L (output layer), so only 1 hidden layer means L=3

		@type s_l : Python list (or tuple, or iterable) of (positive) integers
		@param s_l : list (or tuple or iterable) of length L, containing (positive) integers for s_l, size or number of "nodes" or "units" in layer l=1,2,...L; because of how Python counts from 0, it's indexed 0,1,...L-1
					NOTE that number of "components" of y, K, must be equal to s_L, s_L=K
						
		
		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		"""
		self.L = L 
		self.s_l = s_l
	
		if rng is None:
			rng = np.random.RandomState(1234)
	
		###############
		# BUILD MODEL #
		###############


		Thetabs_lst = [] # temporary list of Thetas, Theta,b weights or parameters
		
		K_in = self.s_l[-1]
		ThetabthetaW1 = ThetabthetaW( (s_l[1],s_l[0]),H= K_in, activation=activation_fxn, l=1, rng=rng)
		ThetabthetaW1.connect_through()
		
		Thetabs_lst.append( ThetabthetaW1 )
		
		for l in range(2,L-1): # don't include the Theta,b going to the output layer in this loop
			inputlayer_al = Thetabs_lst[-1].alp1
			Thetabl = Thetab(l,(s_l[l],s_l[l-1]),al=inputlayer_al, activation=activation_fxn, rng=rng)
			Thetabl.connect_through()
			Thetabs_lst.append( Thetabl )
		
		# (Theta,b), going to output layer, l=L
		if ( L>2):
			inputlayer_al = Thetabs_lst[-1].alp1
			Thetabl = Thetab(L-1,(s_l[L-1],s_l[L-2]),al=inputlayer_al, activation=psi_Lm1, rng=rng)
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
		self.Thetabs[0].h_tm1 = h_tm1
		self.Thetabs[0].c_tm1 = c_tm1
		self.Thetabs[0].connect_through()
		
		for l in range(2,L): # l=2,3,...L-1, for each of the Theta operations between layers l
			self.Thetabs[l-1].al = self.Thetabs[l-2].alp1
			self.Thetabs[l-1].connect_through()
			
		# return the result of the gate (\alpha), denoted h_Theta	
		h_Theta = self.Thetabs[-1].alp1
		return h_Theta
		

class LSTM_Model(object):
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

	NOTES
	=====
	
		
	"""

	def __init__(self, L_in, s_l_in, activations, activation_y, K, rng=None):
		""" Initialize LSTM class
		
		INPUT/PARAMETER(S)
		==================  

		@type L_in : namedtuple of (positive) integers, with entries 'g','i','f','o', in that order
		@param L_in : number of layers each for gates g, i, f, o
		
		@type s_l_in : namedtuple of Python lists (or tuple, or iterable) of (positive) integers
		@param s_l_in : size or number of "nodes" or units for all the layers L^{(\alpha)} for each of the (4) gates, 'g','i','f','o', 
			e.g. \alpha = { 'g','i','o','f' }, let l^{(\alpha})=1,2,...L^{(\alpha)}, then s_l^{\alpha} has the number of nodes or units for that layer l, for that "gate" \alpha
		
		@type activations : namedtuple of tuples, each tuple (of length 2) of theano functions (or numerical numpy functions), with entries 'g','i','f','o','h'
		@param activations : sigmoidal functions necessary for LSTM, denoted psi's

		@type K : (positive) integers
		@param K : number of componenets of output y, y \in \mathbb{R}^K
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
			g= Feedforward( L_in.g, s_l_in.g, activation_fxn= activations.g[0], 
								psi_Lm1=activations.g[-1], rng=rng), 
			i= Feedforward_ifo( L_in.i, s_l_in.i, activation_fxn= activations.i[0], 
								psi_Lm1=activations.i[-1], rng=rng), 
			f= Feedforward_ifo( L_in.f, s_l_in.f, activation_fxn= activations.f[0], 
								psi_Lm1=activations.f[-1], rng=rng), 
			o= Feedforward_ifo( L_in.o, s_l_in.o, activation_fxn= activations.o[0], 
								psi_Lm1=activations.o[-1], rng=rng)  
		)

		H = s_l_in.g[-1] # assert s_l_in.i[-1] == s_l_in.f[-1] == s_l_in.o[-1]
		
		self.Thetaby = Thetab( 1, (K,H), activation=activation_y, rng=rng)

		## unroll all the parameters
		
		gThetas = [Weight.Theta for Weight in self._gates.g.Thetabs]
		gbs     = [Weight.b for Weight in self._gates.g.Thetabs]
		gthetas = [Weight.theta for Weight in self._gates.g.Thetabs]
		self.params = gThetas + gbs + gthetas
		
		iThetas = [Weight.Theta for Weight in self._gates.i.Thetabs]
		ibs     = [Weight.b for Weight in self._gates.i.Thetabs]
		ithetas = [Weight.theta for Weight in self._gates.i.Thetabs]
		iWs     = [Weight.W for Weight in self._gates.i.Thetabs]
		self.params = self.params + iThetas + ibs + ithetas + iWs
		
		fThetas = [Weight.Theta for Weight in self._gates.f.Thetabs]
		fbs     = [Weight.b for Weight in self._gates.f.Thetabs]
		fthetas = [Weight.theta for Weight in self._gates.f.Thetabs]
		fWs     = [Weight.W for Weight in self._gates.f.Thetabs]
		self.params = self.params + fThetas + fbs + fthetas + fWs
		
		oThetas = [Weight.Theta for Weight in self._gates.o.Thetabs]
		obs     = [Weight.b for Weight in self._gates.o.Thetabs]
		othetas = [Weight.theta for Weight in self._gates.o.Thetabs]
		oWs     = [Weight.W for Weight in self._gates.o.Thetabs]
		self.params = self.params + oThetas + obs + othetas + oWs

		self.params = self.params + [ self.Thetaby.Theta, self.Thetaby.b ]

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


class MemoryBlock(object):
	""" MemoryBlock
	
	(AVAILABLE) CLASS MEMBERS
	=========================
	
	"""
	
	def __init__(self, LSTM_model, X=None, h0=None,c0=None):
		""" Initialize MemoryBlock class
		
		INPUT/PARAMETER(S)
		==================  
		@type LSTM_model : LSTM_Model class instance
		@param LSTM_model : LSTM_Model class instance
		
		"""
		self.LSTM_model = LSTM_model

		if X is None:
			X = T.matrix()

		self.X = X

#		H = self.LSTM_model.Thetaby.b.shape[0].astype('int32') # number of components of y, i.e. y\in \mathbb{R}^K
		H = self.LSTM_model.s_l.g[-1]

		self._H = H
		self._c0 = theano.shared(np.zeros((H,1)).astype(theano.config.floatX) )
		self._h0 = T.tanh( self._c0)
		
		
	def build_scan_over_t(self):
		""" build_scan_over_t
		"""
		lstm_step_fxn = self.LSTM_model.build_lstm_step()
		X = self.X
		c0 = self._c0
		h0 = self._h0
		params= self.LSTM_model.params
		
		[h_vals, c_vals, y_vals], updates_from_scan = theano.scan(fn=lstm_step_fxn, 
					sequences=dict(input=X, taps=[0]),
					outputs_info = [h0,c0,None],
					)
#					non_sequences = params)
	
		self.scan_res = [h_vals, c_vals, y_vals], updates_from_scan
		return [h_vals, c_vals, y_vals], updates_from_scan
	

def feedforward_ifg( i_t, f_t, g_t, c_tm1, o_t, Thetaby, psi_h=T.tanh):
	""" feedforward_ifg - "feedforward" for the i, f, g gates, for input, forget, and (actual) input, 
	resulting in updated cell memory value, c_t
	"""
	
	c_t = f_t.alp1 * c_tm1 + i_t.alp1 * g_t.alp1
	
	o_t.c_tm1 = c_t
	o_t.connect_through()
	
	h_t = o_t.alp1 * psi_h( c_t)
	
	Thetaby.al = h_t
	Thetaby.connect_through()
	y_t = Thetaby.alp1

	return [h_t, c_t, y_t]


		
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
	
	LSTM_model_Herta= LSTM_Model( L_Herta, s_l_Herta, activations_Herta, T.nnet.sigmoid, n_y)
	lstm_step_fxn = LSTM_model_Herta.build_lstm_step()
	MemBlck_Herta = MemoryBlock(LSTM_model_Herta)


"""
	rng_test=  np.random.RandomState(1234)
	X_in_sym_test = T.matrix(theano.config.floatX)
	Thetabtheta1_test = Thetabtheta((6,5),al=X_in_sym_test,activation=T.nnet.sigmoid,rng=rng_test)
	Thetabtheta1_test.connect_through()
	Thetab2=Thetab(2,(4,5),al=Thetabtheta1_test.alp1,activation=None,rng=rng_test)
"""	
	
			
			
