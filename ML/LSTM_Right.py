"""
@file LSTM_Right.py
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

		lin_zlp1 = T.dot( self.al, self.Theta) + self.b
		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )

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

	@type .params   : Python list of theano shared variables, list size (length) of 2
	@param .params  : \Theta^{(l)} or ( \Theta^{(l)} , )
		
	@type .Thetas  : Python list of theano shared variables, list size (length) 1
	@param .Thetas : \Theta^{(l)} or ( \Theta^{(l)}, )


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

		lin_zlp1 = T.dot( self.al, 
						self.Theta) + self.b + T.dot( self._h, self.theta)

		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )

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
		theta = self.theta
				
		Thetas = [ Theta, theta]
		bs     = [ b, ]
		params  = [ Theta, b, theta ]
		
		return dict(Thetas=Thetas,bs=bs, params=params)

	def __set_state__(self,*args):
		""" __set_state__
		"""
		Theta_in = args[0]
		b_in     = args[1]
		theta_in = args[2]

		self.Theta.set_value(Theta_in)
		self.b.set_value(b_in)
		self.theta.set_value(theta_in)



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

		lin_zlp1 = T.dot( self.al, 
						self.Theta) + self.b + T.dot( self._h, 
													self.theta) + T.dot( self._c, self.W)

		if self.psi is None:
			self.alp1 = lin_zlp1
		else:
			self.alp1 = self.psi( lin_zlp1 )

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
		theta = self.theta
		W  = self.W
				
		Thetas = [ Theta, theta,W]
		bs     = [ b, ]
		params  = [ Theta, b, theta,W ]
		
		return dict(Thetas=Thetas,bs=bs, params=params)

	def __set_state__(self,*args):
		""" __set_state__
		"""
		Theta_in = args[0]
		b_in     = args[1]
		theta_in = args[2]
		W_in     = args[3]

		self.Theta.set_value(Theta_in)
		self.b.set_value(b_in)
		self.theta.set_value(theta_in)
		self.W.set_value(W_in)



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

			#initialize an instance of class Thetab_right
			Thetabl = Thetab_right(l,(s_l[l-1],s_l[l]),al=inputlayer_al, activation=activation_fxn, rng=rng)
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
		# h_Theta = self.Thetabs[-1].alp1 # output layer, which should be given the values for g_t for LSTM

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

	def __get_state__(self):
		""" __get_state__ - return the parameters or "weights" that were used in this feedforward
		
		"""
		Thetabs = self.Thetabs

		Thetas = [theta for Weight in Thetabs for theta in Weight.__get_state__()['Thetas'] ]
		bs     = [b for Weight in Thetabs for b in Weight.__get_state__()['bs'] ]
		params = [weight for Weight in Thetabs for weight in Weight.__get_state__()['params']]

		
		return dict(Thetas=Thetas,bs=bs,params=params)
		
	def __set_state__(self,*args):
		""" __set_state__
		
		@type *args : expect a flattened out Python list of Theta* classes
		@param *args : use the "power" of *args so we're flexible about Python function argument inputting (could be a list, could be separate entries)
		"""
		L = self.L
		number_of_weights_1 = len(self.Thetabs[0].__get_state__()["params"]) # number of weights related to the 1st Thetab

		Thetabtheta1_vals=[args[w] for w in range(number_of_weights_1)]				

		self.Thetabs[0].__set_state__(*Thetabtheta1_vals)

		flattened_index = number_of_weights_1-1 # count from this value
		for l in range(2,L):  # l=2,...L corresponds to index idx=1,...L-1 (Python counts from 0;I know, it's confusing)
			number_of_weightsl = len(self.Thetabs[l-1].__get_state__()['params'])# number of weights in layer l
			Thetab_vals=[]
			for w in range(number_of_weightsl):
				Thetab_vals.append( args[w+1+flattened_index] )
			
			self.Thetabs[l-1].__set_state__(*Thetab_vals)
			flattened_index += number_of_weightsl
				


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
		# h_Theta = self.Thetabs[-1].alp1 # output layer, which should be given the values for g_t,i_t,f_t for LSTM

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

	def __get_state__(self):
		""" __get_state__ - return the parameters or "weights" that were used in this feedforward
		
		"""
		Thetabs = self.Thetabs
		Thetas = [theta for Weight in Thetabs for theta in Weight.__get_state__()['Thetas'] ]
		bs     = [b for Weight in Thetabs for b in Weight.__get_state__()['bs'] ]
		params = [weight for Weight in Thetabs for weight in Weight.__get_state__()['params']]
		
		return dict(Thetas=Thetas,bs=bs,params=params)
		
		
	def __set_state__(self,*args):
		""" __set_state__
		
		@type *args : expect a flattened out Python list of Theta* classes
		@param *args : use the "power" of *args so we're flexible about Python function argument inputting (could be a list, could be separate entries)
		"""
		L = self.L
		number_of_weights_1 = len(self.Thetabs[0].__get_state__()["params"]) # number of weights related to the 1st Thetab

		Thetabtheta1_vals=[args[w] for w in range(number_of_weights_1)]				

		self.Thetabs[0].__set_state__(*Thetabtheta1_vals)

		flattened_index = number_of_weights_1-1 # count from this value
		for l in range(2,L):  # l=2,...L corresponds to index idx=1,...L-1 (Python counts from 0;I know, it's confusing)
			number_of_weightsl = len(self.Thetabs[l-1].__get_state__()['params'])# number of weights in layer l
			Thetab_vals=[]
			for w in range(number_of_weightsl):
				Thetab_vals.append( args[w+1+flattened_index] )
			
			self.Thetabs[l-1].__set_state__(*Thetab_vals)
			flattened_index += number_of_weightsl	
		
		

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

	
	def __get_state__(self):
		""" __get_state__ - return the parameters or "weights" that were used in this feedforward
		
		"""
		## unroll all the parameters
		gates = self._gates
		Thetaby = self.Thetaby
		

		Thetas = [theta for gate in gates for theta in gate.__get_state__()['Thetas']] + Thetaby.__get_state__()['Thetas']
		params = [weight for gate in gates for weight in gate.__get_state__()['params']] + Thetaby.__get_state__()['params']

		print "Total number of parameters: %d " % len(params) 

		return dict(Thetas=Thetas,params=params)

	def __set_state__(self,*args):
		""" __set_state__ = __set_state__(self,*args)
		
		@type *args : expect a flattened out Python list of Theta* classes
		@param *args : use the "power" of *args so we're flexible about Python function argument inputting (could be a list, could be separate entries)
		"""
		number_of_gates = len(self._gates)
		flattened_index = 0
		for alpha_idx in range(number_of_gates):  # alpha_idx = 0,1,..number_of_gates-1
			number_of_weights_alpha_idx = len( self._gates[alpha_idx].__get_state__()["params"])
			Thetabs_vals=[]
			for w in range(number_of_weights_alpha_idx): # w=0,1,..number_of_weights_alpha_idx-1
				Thetabs_vals.append( args[ flattened_index + w] )

			self._gates[alpha_idx].__set_state__( *Thetabs_vals )
			flattened_index += number_of_weights_alpha_idx


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

		params= self.LSTM_model.__get_state__()['params']
	
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
		Thetas_only = self.LSTM_model.__get_state__()['Thetas']

		
		lambda_val = np.float32( lambda_val )  # regularization constant
		J = build_cost_functional( lambda_val, 
									self.scan_res[0][-1], # we want y_vals from above, predicted value for y
									y_sym, Thetas_only)
		self.J_Theta = J
		return J

	def build_J_L2norm(self, lambda_val):
		""" build_J_L2norm - build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)
		"""
		y_sym = self.y
		Thetas_only = self.LSTM_model.__get_state__()['Thetas']

		
		lambda_val = np.float32( lambda_val )  # regularization constant
		J = build_cost_functional_L2norm( lambda_val, 
									self.scan_res[0][-1], # we want y_vals from above, predicted value for y
									y_sym, Thetas_only)
		self.J_Theta = J
		return J

	def build_update(self,alpha=0.01,beta=0.0):
		""" build_update - build the update step
		
		INPUTS/PARAMETERS
		=================
		@type alpha : float
		@param alpha : learning rate 
		
		@type beta : float
		@param beta : "momentum" constant (parameter)
		
		"""
		J = self.J_Theta
		Thetabs = self.LSTM_model.__get_state__()['params']

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
		
	def prediction_fxn_givens(self,X_vals):
		""" prediction_fxn_given - same as prediction_fxn, result in a theano function, but using givens
		"""
		X=self.X
		y_predicted = sandbox.cuda.basic_ops.gpu_from_host(
						self.scan_res[0][-1]
							)
		predictions = theano.function([],
								outputs = y_predicted,
								givens = { X : X_vals.astype(theano.config.floatX) })
		return predictions
							
		

	def predict_on_lst(self, test_data, verbose=True):
		"""
		@type test_data : Python list (of length m, number of training examples)

		@type verbose : bool
		@param verbose : boolean flag saying to do the print outs (be "verbose"; True) or not (False)
		"""

		predictions = []
		
		for i,o in test_data:  # test_data is a list from j=0,1,...m-1, m total training data points
			predictions_func = self.prediction_fxn()
			i = i.astype(theano.config.floatX)
			predicted_y = predictions_func(i)  # i is a list of arrays of size d features, list of length T, for time t=0,1...T-1

		# EY : 20170223 Try this in the future; uncomment it out
#			i = np.array(i).astype(theano.config.floatX)
#			predicted_y = predictions_func(i)
			
			if verbose:
				print o[-2] # target
				print predicted_y[-2] # prediction
			
			predictions.append( predicted_y )
			
		return predictions
	
	def predict_on_lst_givens(self, test_data, verbose=False):
		""" predict_on_lst_givens - same as predict_on_lst, but using givens for theano function through prediction_fxn_givens
		@type test_data : Python list (of length m, number of training examples)

		@type verbose : bool
		@param verbose : boolean flag saying to do the print outs (be "verbose"; True) or not (False)
		"""

		predictions = []
		
		for i,o in test_data:  # test_data is a list from j=0,1,...m-1, m total training data points
			i = i.astype(theano.config.floatX)  # Txd or (T,d) size dims. matrix
			predictions_func = self.prediction_fxn_givens(i)
			predicted_y = predictions_func()  # i is a list of arrays of size d features, list of length T, for time t=0,1...T-1

			if verbose:
				print o[-2] # target
				print predicted_y[-2] # prediction
			
			predictions.append( predicted_y )
			
		return predictions
	
	
	def calc_J(self,X_vals,y_vals):
		""" calc_J
		"""
		X=self.X
		y=self.y
		J = self.J_Theta
		calculated_J = theano.function([], 
							outputs = J,
							givens = { X : X_vals.astype(theano.config.floatX),
										y : y_vals.astype(theano.config.floatX) } )
		return calculated_J()
	
	
	#################################
	## for Long-Term Serialization ##
	## cf. http://deeplearning.net/software/theano/tutorial/loading_and_saving.html#long-term-serialization
	#################################
	
	def __get_state__(self):
		""" __getstate__(self)  
		
		returns a Python list of numpy arrays, (through theano.shared.get_value() )
		"""
		
		params = self.LSTM_model.__get_state__()['params']
		params_vals = [weight.get_value() for weight in params]
		return params_vals
		
	def __set_state__(self, *args):
		""" __setstate__(self,*args)
		
		@type *args : expect a flattened out Python list of Theta* classes
		@param *args : use the "power" of *args so we're flexible about Python function argument inputting (could be a list, could be separate entries)
		"""
		
		self.LSTM_model.__set_state__( *args)

			
	def save_parameters(self, filename='objects.save'):
		""" save_parameters (save the weights or parameters of this model as numpy arrays that are pickled)
		
		@type filename : string
		@param filename : string to name the saved file
		"""	
		f = open(filename,'wb')
		for param in self.LSTM_model.__get_state__()['params']:
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
	
	@type h     : theano shared variable of size dims. (K,m) (size dim. might be (m,K) due to right action
	@param h    : hypothesis

	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the layers l=1,2,...L-1
	NOTE: remember, we want a list of theano MATRICES, themselves, not the class

	RETURN/OUTPUTS
	==============
	@type J_theta : theano symbolic expression

	"""
	m = y_sym.shape[1].astype(theano.config.floatX)

		# logistic regression cost function J, with no regularization (yet)
	J_theta = - T.mean( y_sym * T.log(h) + (np.float32(1.) - y_sym)*T.log( np.float32(1.) - h) )

	reg_term = np.float32(lambda_val/ (2. )) /m *T.sum( [ T.sum( Theta*Theta) for Theta in Thetas] )

	J_theta = J_theta + reg_term 
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
	m = y_sym.shape[0].astype(theano.config.floatX)

	J_theta = np.float32(0.5) * T.mean( (h - y_sym )*(h-y_sym))

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
	updateThetabs = [ sandbox.cuda.basic_ops.gpu_from_host( 
					Theta - np.float32( alpha) * T.grad( J, Theta) + np.float32(beta)*Theta ) for Theta in Thetabs]
	

	gradientDescent_step = theano.function(inputs = [X_sym, y_sym],
											outputs = J, 
											updates = zip(Thetabs,updateThetabs) )

	return updateThetabs, gradientDescent_step


########################################
##########  __main__  		  ##########
########################################

if __name__ == "__main__":
	print("In main.")

	L = Gates(g=2,i=2,f=2,o=2)
	
	n_hidden = n_i = n_c = n_o = n_f = 10
	n_in = 7 # for embedded reber grammar
	n_y = 7 # for embedded reber grammar; this is K in my notation
	
	s_l = Gates(g=[n_in,n_c],i=[n_in,n_i],f=[n_in,n_f],o=[n_in,n_o])

	activations = Psis(g=(T.tanh, T.tanh), 
							i=(T.nnet.sigmoid, T.nnet.sigmoid),
							f=(T.nnet.sigmoid, T.nnet.sigmoid),
							o=(T.nnet.sigmoid, T.nnet.sigmoid),
							h=(T.tanh,)
							)

	# sigma_y = T.nnet.sigmoid in this case
	LSTM_model= LSTM_Model_right( L, s_l, n_hidden, n_y, activations, T.nnet.sigmoid)
	lstm_step_fxn = LSTM_model.build_lstm_step()
	MemBlck = MemoryBlock_right(n_hidden, LSTM_model)
	MemBlck.build_scan_over_t()
	MemBlck.build_J(0.1)

	# this may take a while below, so I commented it out; uncomment to actually do it
#	MemBlck.build_update()

	import sys
	sys.path.append('../')
	try:
		import reberGrammar
		train_data = reberGrammar.get_n_embedded_examples(1000)
	except ImportError:
		print "Import error, and so we need to locate the file path, manually \n"

	theano.config.allow_gc = False
#	MemBlck.train_model_full(train_data,2)







