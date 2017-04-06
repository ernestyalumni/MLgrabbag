"""
@file GRUs_Right.py
@brief GRU, Gates Recurrent Unit, with mutations

@author Ernest Yeung
@email ernestyalumni dot gmail dot com

cf. Reference paper(s):
http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
An Empirical Exploration of Recurrent Network Architectures. Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever 
"""  
import theano
import numpy as np
import theano.tensor as T
from theano import sandbox

from six.moves import cPickle

from collections import namedtuple

""" Gates - namedtuple (from collections Python library) for the various "gates" needed
	'z' - 
	'r' - 
	'h' - 
	'y' - 
"""
Gates = namedtuple("Gates",['z','r','h','y'])

""" Psis - namedtuple (from collections Python library) for the various "activation" functions needed
	'z' - 
	'r' - 
	'h' - 
	'y' - 
"""
Psis = namedtuple("Psis",['z','r','h','y'])

########################################################################
### Thetab_right class #################################################
########################################################################

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
			b_values =  np.ones(s_lp1).astype(theano.config.floatX)
			b= theano.shared(value=b_values, name='b'+str(l), borrow=True)	
			
		if al is None:
			al = T.matrix(dtype=theano.config.floatX)
#			al = T.row(dtype=theano.config.floatX)
			
			
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
### Thetabtheta_right class ############################################
########################################################################

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
#			al = T.row(dtype=theano.config.floatX)
		
		
		if _h is None:
			_h = T.matrix(dtype=theano.config.floatX)	
#			_h = T.row(dtype=theano.config.floatX)	

			
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
	

	def connect_through(self, al_in=None, h_in=None):
		""" connect_through

			Note that I made connect_through a separate class method, separate from the automatic initialization, 
			because you can then make changes to the "layer units" or "nodes" before "connecting the layers"
				"""
		if al_in is not None:
			self.al = al_in
		if h_in is not None:
			self._h = h_in

		lin_zlp1 = T.dot( self.al, 
						self.Theta) + self.b + T.dot( self._h, self.theta)

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

########################################################################
### END of Thetabtheta_right class #####################################
########################################################################


########################################################################
## Feedforward_y_right class ###########################################
########################################################################


class Feedforward_y_right(object):
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
		Thetab1 = Thetab_right(1, (s_l[0],s_l[1]), activation=activation_fxn, rng=rng)
		Thetab1.connect_through()
		
		Thetabs_lst.append( Thetab1 )
		
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

	def connect_through(self, X_t):
		""" connect_through - connect through the layers (the actual feedforward operation)

		INPUTS/PARAMETERS
		=================
		@type X_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		@type y_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		"""
		self.Thetabs[0].al = X_t
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

		Thetab1_vals=[args[w] for w in range(number_of_weights_1)]				

		self.Thetabs[0].__set_state__(*Thetab1_vals)

		flattened_index = number_of_weights_1-1 # count from this value
		for l in range(2,L):  # l=2,...L corresponds to index idx=1,...L-1 (Python counts from 0;I know, it's confusing)
			number_of_weightsl = len(self.Thetabs[l-1].__get_state__()['params'])# number of weights in layer l
			Thetab_vals=[]
			for w in range(number_of_weightsl):
				Thetab_vals.append( args[w+1+flattened_index] )
			
			self.Thetabs[l-1].__set_state__(*Thetab_vals)
			flattened_index += number_of_weightsl
							

########################################################################
## END of Feedforward_y_right class ####################################
########################################################################

########################################################################
# GRU_MUT001_right class ###############################################
########################################################################

class GRU_MUT001_right(object):
	""" GRU_MUT001_right - represents the GRU (Gated Recurrent Unit), with mutation 1, model, with its z,r "gates"

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

	def __init__(self, L_in, s_l_in, H,K,activations, rng=None):
		""" Initialize LSTM class
		
		INPUT/PARAMETER(S)
		==================  

		@type L_in : namedtuple of (positive) integers, with entries 'z','r','h','y', in that order
		@param L_in : number of layers each for gates z, r, h, y
		
		@type s_l_in : namedtuple of Python lists (or tuple, or iterable) of (positive) integers
		@param s_l_in : size or number of "nodes" or units for all the layers L^{(\alpha)} for each of the (4) gates, 'z','r','h','y', 
			e.g. \alpha = { 'z','r','h','y' }, let l^{(\alpha})=1,2,...L^{(\alpha)}, then s_l^{\alpha} has the number of nodes or units for that layer l, for that "gate" \alpha

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
		
		# ._gates contains the 'z','r','h','y' gates
		self._gates = Gates( 
			z = Thetab_right( 1, s_l_in.z, activation=activations.z[0], rng=rng),
			r = Thetabtheta_right( s_l_in.r, H, activation=activations.r[0], l=1, rng=rng),
			h = Thetab_right( 1, s_l_in.h, activation=activations.h[0], rng=rng),
			y = Feedforward_y_right( L_in.y, s_l_in.y, H,activation_fxn=activations.y[0], psi_Lm1=activations.y[-1],rng=rng)
		)
		
		self._gates.z.connect_through()
		self._gates.r.connect_through()
		self._gates.h.connect_through()



	def build_gru_mut001_step(self):
		""" build_gru_mut001_step - returns a function, named gru_mut001_step, that executes (1) GRU MUT1 step
		
		gru_mut001_step = gru_mut001_step(X_t, h_tm1)
		"""

		def gru_mut001_step(X_t, h_tm1, *args_for_params): 

			z_t = self._gates.z.connect_through(X_t)
			r_t = self._gates.r.connect_through(X_t, h_tm1)
			
#			h_t = self._gates.h.connect_through( r_t * h_tm1)
			h_t = self._gates.h.connect_through( r_t)

			h_t = h_t + self.psis.h[0]( self.psis.h[0]( X_t))
			h_t = h_t * z_t  + h_t * (np.cast[theano.config.floatX](1.) - z_t)

			y_t = self._gates.y.connect_through( h_t)
			y_t = sandbox.cuda.basic_ops.gpu_from_host( y_t )
			
			return [h_t, y_t]
	
		return gru_mut001_step

	
	def __get_state__(self):
		""" __get_state__ - return the parameters or "weights" that were used in this feedforward
		
		"""
		## unroll all the parameters
		gates = self._gates
		
		Thetas = [theta for gate in gates for theta in gate.__get_state__()['Thetas']]  
		params = [weight for gate in gates for weight in gate.__get_state__()['params']]

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
	def __init__(self, H, GRU_model, X=None, y=None,h0=None):
		""" Initialize MemoryBlock class
		
		INPUT/PARAMETER(S)
		==================  
		@type H : (positive) integers
		@param H : number of componenets of "hidden layer units" h_t, h_t \in \mathbb{R}^H

		@type GRU_model : GRU_MUT001 class instance
		@param GRU_model : GRU_MUT001 class instance
		
		"""
		self.GRU_model = GRU_model

		if X is None:
			X = T.matrix(dtype=theano.config.floatX)
#			X = T.row(dtype=theano.config.floatX)

		self.X = X

		if y is None:
			y = T.matrix(dtype=theano.config.floatX)
#			y = T.row(dtype=theano.config.floatX)

		self.y =y

		if h0 is None:
			h0 = theano.shared(np.zeros(H).astype(theano.config.floatX) )
	
		self._h0 = h0	
		
	def build_scan_over_t(self,X=None):
		""" build_scan_over_t
		"""
		if X is not None:
			self.X = X
		else:
			X = self.X

		h0 = self._h0
		gru_mut001_step_fxn = self.GRU_model.build_gru_mut001_step()

		params= self.GRU_model.__get_state__()['params']
		

		[h_vals, y_vals], updates_from_scan = theano.scan(fn=gru_mut001_step_fxn, 
					sequences=dict(input=X, taps=[0]),
					outputs_info = [h0,None],
					non_sequences = params)
				
	
		self.scan_res = [h_vals, y_vals], updates_from_scan
		return [h_vals, y_vals], updates_from_scan

	def build_J(self, lambda_val):
		""" build_J - build or make cost functional
		"""
		y_sym = self.y
		Thetas_only = self.GRU_model.__get_state__()['Thetas']
		
		lambda_val = np.float32( lambda_val )  # regularization constant
		J = build_cost_functional( lambda_val, 
									self.scan_res[0][-1], # we want y_vals from above, predicted value for y
									y_sym, Thetas_only)
		self.J_Theta = J
		return J

	def build_J_L2norm(self, lambda_val, y_sym=None):
		""" build_J_L2norm - build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)
		"""
		if y_sym is not None:
			self.y = y_sym
		else:
			y_sym = self.y
		
		Thetas_only = self.GRU_model.__get_state__()['Thetas']
		
		lambda_val = np.cast[theano.config.floatX]( lambda_val )  # regularization constant
		J = build_cost_functional_L2norm( lambda_val, 
									self.scan_res[0][-1], # we want y_vals from above, predicted value for y
									y_sym, Thetas_only)

		J = sandbox.cuda.basic_ops.gpu_from_host( J )
		
		self.J_Theta = J
		return J

	def build_update(self,alpha=0.01,beta=0.0,X=None,y=None):
		""" build_update - build the update step
		
		INPUTS/PARAMETERS
		=================
		@type alpha : float
		@param alpha : learning rate 
		
		@type beta : float
		@param beta : "momentum" constant (parameter)
		
		"""
		J = self.J_Theta
		Thetabs = self.GRU_model.__get_state__()['params']

		if X is not None:
			self.X = X
		else:
			X = self.X
		if y is not None:
			self.y = y
		else:
			y = self.y
		
		self.updateExpression, self.gradDescent_step = build_gradDescent_step(J, Thetabs,X,y,alpha,beta)

	
	###############
	# TRAIN MODEL #
	###############	
		
	def train_model_full(self, train_data, max_iters=250):
		print "theano.config.allow_gc =: " , theano.config.allow_gc

		train_errors = np.ndarray(max_iters)
#		train_errors = []
		
		learn_grad = self.gradDescent_step
		
		m = len(train_data)
		
		for iter in range( max_iters) :
			error = 0.
#			error = []

			for j in range(m):
				X,y = train_data[j]
				X=X.astype(theano.config.floatX)
				y=y.astype(theano.config.floatX)
				J_train = learn_grad(X,y) # cost of training, train_cost
				
#				error.append( J_train)
				if np.isnan( J_train ) or np.isinf( J_train):
#					print('bad cost detected: ', J_train )
					error += np.array( 1.)
				else: 
					error += np.array( J_train)
					
			train_errors[iter] = error
#			train_errors.append( error )

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
							
		

	def predict_on_lst(self, test_data, verbose=False):
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
		
		params = self.GRU_model.__get_state__()['params']
		params_vals = [weight.get_value() for weight in params]
		return params_vals
		
	def __set_state__(self, *args):
		""" __setstate__(self,*args)
		
		@type *args : expect a flattened out Python list of Theta* classes
		@param *args : use the "power" of *args so we're flexible about Python function argument inputting (could be a list, could be separate entries)
		"""
		
		self.GRU_model.__set_state__( *args)

			
	def save_parameters(self, filename='objects.save'):
		""" save_parameters (save the weights or parameters of this model as numpy arrays that are pickled)
		
		@type filename : string
		@param filename : string to name the saved file
		"""	
		f = open(filename,'wb')
		for param in self.GRU_model.__get_state__()['params']:
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
	m = y_sym.shape[0].astype(theano.config.floatX)

		# logistic regression cost function J, with no regularization (yet)
	J_theta = T.sum( T.nnet.categorical_crossentropy( h, y_sym ) )
	
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
#	m = y_sym.shape[0].astype(theano.config.floatX)

	J_theta = np.cast[theano.config.floatX](0.5) * T.mean( T.sqr(h - y_sym ))

#	reg_term = np.cast[theano.config.floatX](lambda_val/ (2. )) /m *T.sum( [ T.sum( Theta*Theta) for Theta in Thetas] )
#	reg_term = np.cast[theano.config.floatX](lambda_val/ (2. ))  *T.mean( [ T.sum( Theta*Theta) for Theta in Thetas] )

#	reg_term = np.cast[theano.config.floatX](lambda_val/ (2. ))  *T.mean( [ T.sum( Theta*Theta, acc_dtype=theano.config.floatX) for Theta in Thetas], acc_dtype=theano.config.floatX )
	reg_term = np.cast[theano.config.floatX](lambda_val/ (2. ))  *T.mean( [ T.sum( T.sqr(Theta), acc_dtype=theano.config.floatX) for Theta in Thetas], acc_dtype=theano.config.floatX )


#	J_theta = J_theta + reg_term
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



########################################################################
## END of GRU_MUT001 class ####################################
########################################################################

if __name__ == "__main__":
	print("In main")

	L = Gates(z=2,r=2,h=2,y=2)
	H = 3
	n_in = 3
	s_l = Gates(z=[n_in,H],r=[n_in,H],h=[H,H],y=[H,2])
	activations=Psis(z=(T.nnet.sigmoid, T.nnet.sigmoid),
					r=(T.nnet.sigmoid,T.nnet.sigmoid),
					h=(T.tanh,T.tanh),
					y=(None,None))
	
	lambda_learn = 0.01 # regularization for cost function
	alpha_reg = 0.9 # learning rate
	beta_mom   = 0.001 # momentum constant 
					
	GRU_MUT001_model = GRU_MUT001_right( L,s_l,H,2,activations)				
	
	# sanity check print out
	print( theano.pp( GRU_MUT001_model._gates.z.alp1 ) )
	print( theano.pp( GRU_MUT001_model._gates.r.alp1 ) )
	print( theano.pp( GRU_MUT001_model._gates.h.alp1 ) )
	print( theano.pp( GRU_MUT001_model._gates.y.Thetabs[-1].alp1 ) )
	
	GRU_MUT001_model.build_gru_mut001_step()
	
	MemBlck = MemoryBlock_right(H,GRU_MUT001_model)
	MemBlck.build_scan_over_t()
	MemBlck.build_J_L2norm(lambda_learn)
	MemBlck.build_update(alpha=alpha_reg,beta=beta_mom)


