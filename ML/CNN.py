"""
	@file   : CNN.py
	@brief  : Convolution Neural Network with theano
	@author : Ernest Yeung	ernestyalumni@gmail.com
	@date   : 20170802
	@ref    :  cf. https://github.com/ernestyalumni/MLgrabbag/tree/master/ML  
  
	If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
	
	https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
	
	which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
	Otherwise, I receive emails and messages on how all my (free) material on 
	physics, math, and engineering have helped students with their studies, 
	and I know what it's like to not have money as a student, but love physics 
	(or math, sciences, etc.), so I am committed to keeping all my material 
	open-source and free, whether or not 
	sufficiently crowdfunded, under the open-source MIT license: 
	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
	Just don't be an asshole and not give credit where credit is due.  
	Peace out, never give up! -EY
"""
import theano
import numpy as np
import theano.tensor as T
from theano import sandbox
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

import DNN
from DNN import Axon

########################################################################
### Convolution axon (as right module) class ###########################
########################################################################

class Axon_CNN(object):
	""" 
	@class Axon_CNN 
	@brief Convolution Axon class, for parameters or weights and intercepts b between layers l-1 and l, for right action, 
	i.e. the matrix (action) multiplies from the right onto the vector (the module or i.e. vector space)

	(AVAILABLE) CLASS MEMBERS
	=========================
	@type .c  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_lxs_{l+1}), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
	@param .c : "weights" or parameters for l, l=1,2, ... L-1
		
	@type .b 	  : theano shared of dim. s_{l+1} or s_lp1
	@param .b     : intercepts

	@type .alm1 	  : theano shared variable or vector of size dims. (m,s_l), m=1,2..., number of training examples
	@param .alm1    : "nodes" or "units" of layer l

	@type .al   : theano symbolic expression for array of size dims. (m,s_lp1)
	@param .al  : "nodes" or "units" of layer l+1

	@type .l      : (positive) integer l=0,1,2,..L-1, L+1 is total number of layers, and l=L is "output" layer
	@param .l     : which layer we're "starting from" for Theta^{(l)}, l=0,1,...L-1

	@type .psi      : function, could be theano function
	@param .psi     : function (e.g. such as sigmoid, tanh, softmax, etc.)


	NOTES
	=====
	borrow=True GPU is ok
	cf. http://deeplearning.net/software/theano/tutorial/aliasing.html

	after initialization (automatic, via __init__), remember to "connect the layers l-1 and l" with the class method connect_through
	"""

	def __init__(self, l, C_ls, Wl, Pl=None, alm1=None, c=None, b=None, activation=T.tanh, rng=None ):
		""" Initialize the parameters for the `layer`

		@type rng  : numpy.random.RandomState
		@param rng : random number generator used to initialize weights
		
		@type l    : (positive) integer
		@param l   : layer number label, l=0 (input),1,...L-1, L is "output layer"

		@type Cl : tuple of (positive) integers of size (length) 2, only
		@param Cls : "matrix" size dimensions of Theta or weight mmatrix, of size dims. (s_l, s_lp1) (this is important)
							Bottom line: s_ls = (s_l, s_lp1)

		@type alm1     : theano shared variable or vector of size dims. (m, s_l), m=1,2..., number of training examples
		@oaram alm1    : "nodes" or "units" of layer l

		@type c  : theano shared; of size dims. (s_l,s_{l+1}), i.e. matrix dims. (s_{l}x(s_{l+1})), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_l,s_{l+1}) 
		@param c : "weights" or parameters for l, l=1,2, ... L-1
		
		@type b      : theano shared of dim. s_{l+1} or s_lp1; it's now a "row" array of that length s_lp1
		@param b     : intercepts

		@type activation  : theano.Op or function
		@param activation : Non linearity to be applied in the layer(s)
		"""
		C_lm1, C_l = C_ls

		if rng is None:
			rng = np.random.RandomState(1234)
	
		# num input feature maps * filter height * filter width 
		fan_in = C_lm1 * np.prod( Wl)
	
		# num output feature maps * filter height * filter width / pooling size 
		if Pl is not None:
			fan_out = C_l * np.prod(Wl) // np.prod(Pl)
		else:
			fan_out = C_l * np.prod(Wl) 
			

		# make the filter size out of C_ls and Wl
		filter_size = (C_l,C_lm1) + Wl
		assert len(filter_size) == (2+len(Wl))

		if c is None:
			try:
				c_values = np.asarray( 
					rng.uniform( 
						low=-np.sqrt(6. / ( fan_in + fan_out )), 
						high=np.sqrt(6. / ( fan_in + fan_out )), size=filter_size  ), 
						dtype=theano.config.floatX 
				)
				
			except MemoryError:
				c_values = np.zeros(filter_size).astype(theano.config.floatX)	
			
				
			if activation == T.nnet.sigmoid:
				c_values *= np.float32( 4 )
			
			c = theano.shared(c_values, name="c"+str(l), borrow=True)
		

		if b is None:
			b_values =  np.zeros((C_l,)).astype(theano.config.floatX)
			b= theano.shared(value=b_values, name='b'+str(l), borrow=True)	
			
		if alm1 is None:
			alm1 = T.tensor4(name='a'+str(l)+'m1',dtype=theano.config.floatX)
			
	
		self.c = c  # size dims. (C_l,C_lm1,W_1,...W_d) i.e. C_l x C_lm1 x W_1,x ... x W_d
		self.b     = b      # dims. C_l
		self.alm1    = alm1     # dims. (m,C_lm1,L_1,...L_d)
		self.C_ls = C_ls
		self.Wl = Wl
		self.Pl = Pl
		
		self.l     = l

		if activation is None:
			self.psi = None
		else:
			self.psi = activation

		# do a "basic" convolution, do a "basic" connect through
		conv_out = conv2d(self.alm1, self.c)
		
		if Pl is not None:
			pooled_out = pool.pool_2d(conv_out, self.Pl, ignore_border=True)
			if self.psi is None:
				self.al = pooled_out + self.b.dimshuffle('x',0,'x','x')
			else:
				self.al = self.psi( pooled_out + self.b.dimshuffle('x',0,'x','x') )
		else:
			if self.psi is None:
				self.al = conv_out + self.b.dimshuffle('x',0,'x','x')
			else:
				self.al = self.psi( conv_out + self.b.dimshuffle('x',0,'x','x') )
			


	def connect_through(self, alm1_in=None, Pl=None, Ll=None):
		""" connect_through

			@note Note that I made connect_through a separate class method, separate from the automatic initialization, 
			because you can then make changes to the "layer units" or "nodes" before "connecting the layers"
				"""

		if alm1_in is not None:
			self.alm1 = alm1_in

		
		self.Pl = Pl
		alm1 = self.alm1
		c = self.c
		C_lm1,C_l = self.C_ls 
		Wl = self.Wl
		filter_shape = (C_l,C_lm1)+Wl
		assert len(filter_shape) == (2+len(Wl))

		# convolve input feature maps with filters
		if Ll is not None:

			batch_size = alm1.shape[0] # This is m, number of input examples
#			image_shape = (batch_size,C_lm1)+Ll	
			image_shape = (None,C_lm1)+Ll	
			conv_out = conv2d( 
						input=alm1, 
						filters=c, 
						filter_shape=filter_shape,
						input_shape=image_shape)
		else:
			conv_out = conv2d( 
						input=alm1, 
						filters=c, 
						filter_shape=filter_shape)					

		# pool each feature map individually, using maxpooling 
		if Pl is not None:
			pooled_out = pool.pool_2d(
						input=conv_out,
						ws=Pl,
						ignore_border=True)
		
			# add bias term
			if self.psi is None:
				self.al = pooled_out + self.b.dimshuffle('x',0,'x','x')
			else:
				self.al = self.psi( pooled_out + self.b.dimshuffle('x',0,'x','x') )
		else:
			# add bias term
			if self.psi is None:
				self.al = conv_out + self.b.dimshuffle('x',0,'x','x')
			else:
				self.al = self.psi( conv_out + self.b.dimshuffle('x',0,'x','x') )
			

		return self.al

	def __get_state__(self):
		""" __get_state__ 

		This method was necessary, because 
		this is how our Theta/Layer combo/class, Thetab_right, INTERFACES with Feedforward

		OUTPUT(S)/RETURNS
		=================
		@type : Python dictionary 
		"""  
		c = self.c
		b = self.b
				
		cs = [ c, ]
		bs     = [ b, ]
		params = [ c, b]
		
		return dict(Thetas=cs,bs=bs,params=params)
		
	def __set_state__(self,*args):
		""" __set_state__
		"""
		c_in = args[0]
		b_in     = args[1]
		self.c.set_value(c_in)
		self.b.set_value(b_in)


########################################################################
### END of CNN Axon (as right module) class ############################
########################################################################

# dim_data

C_dim_data_template = { "C_ls" : (1,1), "Wl" : (1,1), 
						"Pl" : (2,2), "Ll" : (1,1) }


########################################################################
## Feedforward class for CNN ###########################################
########################################################################

class Feedforward(object):
	""" Feedforward - Feedforward
	"""
	def __init__(self, L, CorD, dim_data, activation_fxn=T.tanh, psi_L=T.tanh, rng=None ):
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
		self._CorD = CorD
		self._dim_data = dim_data
	
		if rng is None:
			rng = np.random.RandomState(1234)
	
		###############
		# BUILD MODEL #
		###############

		Axons_lst = [] # temporary list of axons

		# initialize an instance of class Axon
		if CorD[0] is "C":
			C_0,C_1=dim_data[0]['C_ls']
			W1 = dim_data[0]['Wl']
			P1 = dim_data[0]['Pl']
			L1=dim_data[0]['Ll']

			Axon1=Axon_CNN(1, (C_0,C_1), W1,Pl=P1,activation=activation_fxn, rng=rng)
		

			Axon1.connect_through(Pl=P1,Ll=L1)
			Axons_lst.append( Axon1)
		elif CorD[0] is "D":
			s_l= dim_data[0]
			Axon1 = Axon(1, (s_l[0],s_l[1]), activation=activation_fxn, rng=rng)
			Axon1.connect_through()
		
			Axons_lst.append( Axon1 )
		
		for l in range(2,L): # don't include the Theta,b going to the output layer in this loop
			inputlayer_alm1 = Axons_lst[-1].al

			# initialize an instance of class Axon
			if CorD[l-1] is "C":
				C_lm1,C_l=dim_data[l-1]['C_ls']
				Wl = dim_data[l-1]['Wl']
				Pl = dim_data[l-1]['Pl']
				Ll = dim_data[l-1]['Ll']

				# sanity dimensions check
				if CorD[l-2] is "C":
					C_lm2,C_lm1_previous =dim_data[l-2]['C_ls']
					Wlm1 = dim_data[l-2]['Wl']
					Plm1 = dim_data[l-2]['Pl']
					Llm1 = dim_data[l-2]['Ll']
				
					try:
						assert( C_lm1_previous == C_lm1)
					except AssertionError as Asserterr:
						print(Asserterr, 
							" C_lm1 is %d while previously, C_lm1_previous is %d, for l : %l " % C_lm1, C_lm1_previous, l)
					
					try:
						assert( len(Wl) == len(Wlm1) )
					except AssertionError as Asserterr:
						print(Asserterr, 
							" Wl, Wlm1 are not the same dimension (size), for Axon l= %d ", l)
					try:
						if Pl is not None and Plm1 is not None:
							assert( len(Pl) == len(Plm1) )
					except AssertionError as Asserterr:
						print(Asserterr, 
							" Pl, Plm1 are not the same dimension (size), for Axon l= %d ", l)
					try:
						if Ll is not None and Llm1 is not None:
							assert( len(Ll) == len(Llm1) )
					except AssertionError as Asserterr:
						print(Asserterr, 
							" Ll, Llm1 are not the same dimension (size), for Axon l= %d ", l)

					if Ll is not None and Llm1 is not None:
						d = len(Llm1)
						for i in range(d):  # i=0,1,...d-1 
							if Plm1 is not None:
#								i_sizedim_after_pooling = (size_dim_Llm1[i]-Wlm1[i]+1) // Plm1[i]
								i_sizedim_after_pooling = (Llm1[i]-Wlm1[i]+1) // Plm1[i]
								try:
									assert(i_sizedim_after_pooling == Ll[i] )
								except AssertionError as Asserterr:
									print(Asserterr,
										" for %d th dim., for Axon %d, we obtain incorrect size dim. after pooling. " % i, l)
							else:
#								i_sizedim_after_filter = (size_dim_Llm1[i]-Wlm1[i]+1) 
								i_sizedim_after_filter = (Llm1[i]-Wlm1[i]+1) 
								try:
									assert(i_sizedim_after_filter == Ll[i] )
								except AssertionError as Asserterr:
									print(Asserterr,
										" for %d th dim., for Axon %d, we obtain incorrect size dim. after filtering. " % i, l)
							
				# END of sanity dimensions check

				Axonl=Axon_CNN(l, (C_lm1,C_l), Wl, Pl=Pl, alm1=inputlayer_alm1,
									activation=activation_fxn, rng=rng)
				Axonl.connect_through(Pl=Pl,Ll=Ll)
				Axons_lst.append( Axonl)
			elif CorD[l-1] is "D":
				# for the case of when we go from a convolution axon to a DNN axon, we have to flatten the input
				if CorD[l-2] is "C": 
					inputlayer_alm1 = inputlayer_alm1.flatten(2)
	
				s_l= dim_data[l-1]
				Axonl = Axon(l, (s_l[0],s_l[1]), alm1=inputlayer_alm1,activation=activation_fxn, rng=rng)
				Axonl.connect_through()
		
				Axons_lst.append( Axonl )

		# "weights" and bias, Axon, going to output layer, l=L
		if (L>1):
			inputlayer_aLm1 = Axons_lst[-1].al
		
			# initialize an instance of Axon class 
			if CorD[L-1] is "C":
				C_Lm1,C_L=dim_data[L-1]['C_ls']
				WL = dim_data[L-1]['Wl']
				PL = dim_data[L-1]['Pl']
				LL = dim_data[L-1]['Ll']

				AxonL=Axon_CNN(L, (C_Lm1,C_L), WL,Pl=PL, alm1=inputlayer_aLm1,
									activation=psi_L, rng=rng)
				AxonL.connect_through(Pl=PL,Ll=LL)
				Axons_lst.append( AxonL)
			elif CorD[L-1] is "D":
				# for the case of when we go from a convolution axon to a DNN axon, we have to flatten the input
				if CorD[L-2] is "C": 
					inputlayer_aLm1 = inputlayer_aLm1.flatten(2)

				s_l = dim_data[L-1]
	
				AxonL = Axon(L,(s_l[0],s_l[1]),alm1=inputlayer_aLm1, activation=psi_L, rng=rng)

				AxonL.connect_through()
					
				Axons_lst.append(AxonL)
		
		self.Axons = Axons_lst
		

	def connect_through(self, X_in=None):
		""" connect_through - connect through the layers (the actual feedforward operation)

		INPUTS/PARAMETERS
		=================
		@type X_in : theano shared variable or theano symbolic variable (such as T.matrix, T.vector) but with values set

		"""
		if X_in is not None:
			self.Axons[0].alm1 = X_in

		dim_data = self._dim_data

		CorD = self._CorD

		# initialize an instance of class Axon
		if CorD[0] is "C":
			C_0,C_1=dim_data[0]['C_ls']
			W1 = dim_data[0]['Wl']
			P1 = dim_data[0]['Pl']
			L1 = dim_data[0]['Ll']

			self.Axons[0].connect_through(Pl=P1,Ll=L1)
		elif CorD[0] is "D":
			self.Axons[0].connect_through()
					
		L = self.L
		
		for idx in range(1,L):  	# idx=1,...L-1, for each of the Theta operations for Axon idx+1
			if CorD[idx] is "C":
				C_lm1,C_l=dim_data[0]['C_ls']
				Wl = dim_data[idx]['Wl']
				Pl = dim_data[idx]['Pl']
				Ll = dim_data[idx]['Ll']

				self.Axons[idx].alm1 = self.Axons[idx-1].al

				self.Axons[idx].connect_through(Pl=Pl,Ll=Ll)
			elif CorD[idx] is "D":
				if CorD[idx-1] is "C":
					self.Axons[idx].alm1 = self.Axons[idx-1].al.flatten(2)
				else:
					self.Axons[idx].alm1 = self.Axons[idx-1].al

				self.Axons[idx].connect_through()
		
		# return the result of Feedforward operation, denoted h_c (hypothesis, given "weights" or filters (stencils) c)
		h_c = self.Axons[-1].al
		return h_c

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

		Axon1_vals=[args[w] for w in range(number_of_weights_1)]				

		self.Axons[0].__set_state__(*Axon1_vals)   

		flattened_index = number_of_weights_1-1 # count from this value
		for l in range(1,L):  # l=1,...L-1 corresponds to index idx=1,...L-1 (Python counts from 0;I know, it's confusing)
			number_of_weightsl = len(self.Axons[l].__get_state__()['params'])# number of weights in layer l
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
		outer_layer = Axons[-1].al
		return outer_layer						

########################################################################
## END of Feedforward class for CNN ####################################
########################################################################

########################################################################
############## Convolution Neural Network CNN class ####################
########################################################################

class CNN(object):
	""" CNN - Deep Neural Network
	
	(AVAILABLE) CLASS MEMBERS
	=========================
	
	"""
	def __init__(self,CNN_model,X=None,y=None,borrow=True):
		""" Initialize MemoryBlock class
		
		INPUT/PARAMETER(S)
		==================  

		@type X : numpy array of size dims. (m,d)
		@param X : input data to train on.  NOTE the size dims. or shape and how it should equal what's inputted in d,m
		
		@type y : numpy array of size dims. (m,K)
		@param y : output data of training examples
		
		USAGE/USING THIS CLASS REMARKS:
		===============================
		While the default values of X and y (representing input data and the actual output values 
		(to test against the calculated, predicted values) are None, and None, respectively, 
		you'd want to load your data, from numpy arrays that reside on the hard drive 
		and then to the RAM, and then load it onto the GPU as a shared variable.  
		You don't want very many memory transfers between CPU and GPU.  
		You want to load as much, 1 time, onto the GPU from the beginning, 
		do all computations on the GPU, and transfer back at the very end.  
		
		i.e. 
		
		The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
		
		So initialize CNN class with your data from the beginning.  
		Be sure the size dimensions ("shapes" are all correct)  
		
		"""
		self._CNN_model = CNN_model

		if X is None:
			self.X = T.matrix(dtype=theano.config.floatX)
		else:
			self.X = theano.shared( X.astype(theano.config.floatX), borrow=borrow)

		if y is None:
			y = T.matrix(dtype=theano.config.floatX)
		else:
			self.y = theano.shared( y.astype(theano.config.floatX), borrow=borrow)

	def connect_through(self,X=None,borrow=True):
		""" connect_through - 
		
		@brief :: connect through the Axons (the actual feedforward operation)

		"""
		if X is not None:
			self.X = theano.shared( X.astype(theano.config.floatX),borrow=borrow)
#		X=self.X
		
		h = self._CNN_model.connect_through(X)
		return h 

	def set_y(self,y_vals,borrow=True):
		""" @fn set_y
		@brief :: set class member variable .y with an inputted numpy array
		"""
		self.y = theano.shared( y_vals.astype(theano.config.floatX),borrow=borrow)
		return self.y

	###############################
	# BUILD cost functions J ######
	###############################
	

	def build_J_L2norm_w_reg(self, lambda_val, y_sym=None):
		""" build_J_L2norm_w_reg - build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)

		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		"""
		if y_sym is not None:
			self.y = y_sym
		else:
			y_sym = self.y
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		
		h = self._CNN_model._get_outer_layer_()
		
		lambda_val = np.cast[theano.config.floatX]( lambda_val )  # regularization constant
		J = build_cost_functional_L2norm_w_reg( lambda_val, 
									h, # we want y_vals from above, predicted value for y
									y_sym, Thetas_only)

		J = sandbox.cuda.basic_ops.gpu_from_host( J )
		
		self.J_Theta = J
		return J

	def build_J_L2norm(self, y_sym=None):
		""" build_J_L2norm - build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)
		"""
		if y_sym is not None:
			self.y = y_sym
		else:
			y_sym = self.y
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		h = self._CNN_model._get_outer_layer_()
		
		J = build_cost_functional_L2norm(h, y_sym, Thetas_only)
		J = sandbox.cuda.basic_ops.gpu_from_host( J )
		self.J_Theta = J
		return J

	def build_J_logistic(self, y_sym=None):
		""" @fn :: build_J_logistic - 
			@brief :: build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)
		"""
		if y_sym is not None:
			self.y = y_sym
		else:
			y_sym = self.y
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		h = self._CNN_model._get_outer_layer_()
		
		J = build_cost_functional_logistic(h, y_sym, Thetas_only)
		J = sandbox.cuda.basic_ops.gpu_from_host( J )
		self.J_Theta = J
		return J

	def build_J_logistic_w_reg(self, lambda_val, y_sym=None):
		""" @fn :: build_J_L2norm_w_reg - 
		@brief :: build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)

		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		"""
		if y_sym is not None:
			self.y = y_sym
		else:
			y_sym = self.y
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		
		h = self._CNN_model._get_outer_layer_()
		
		lambda_val = np.cast[theano.config.floatX]( lambda_val )  # regularization constant
		J = build_cost_functional_logistic_w_reg( lambda_val, 
									h, # we want y_vals from above, predicted value for y
									y_sym, Thetas_only)

		J = sandbox.cuda.basic_ops.gpu_from_host( J )
		
		self.J_Theta = J
		return J


	def build_J_xent(self, y_sym=None):
		""" build_J_L2norm - build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)
		"""
		if y_sym is not None:
			self.y = y_sym
		else:
			y_sym = self.y
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		h = self._CNN_model._get_outer_layer_().flatten()
		
		J = build_cost_functional_xent(h, y_sym, Thetas_only)
		J = sandbox.cuda.basic_ops.gpu_from_host( J )
		self.J_Theta = J
		return J

	def build_J_xent_w_reg(self, lambda_val,y_sym=None):
		""" build_J_L2norm - build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)
		"""
		if y_sym is not None:
			self.y = y_sym
		else:
			y_sym = self.y
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		h = self._CNN_model._get_outer_layer_().flatten()
		
		lambda_val = np.cast[theano.config.floatX]( lambda_val )  # regularization constant

		J = build_cost_functional_xent_w_reg(lambda_val,h, y_sym, Thetas_only)
		J = sandbox.cuda.basic_ops.gpu_from_host( J )
		self.J_Theta = J
		return J		

	###############################
	# BUILD MODEL('S UPDATE STEP) #
	###############################

	def build_update(self, alpha=0.01, beta=0.0):
		"""
		@type alpha : float
		@param alpha : learning rate
		
		@type beta : float
		@param beta : "momentum" parameter (it's in the update step for gradient descent)

		"""
		
		Thetabs = self._CNN_model.__get_state__()['params']
		J = self.J_Theta

		
		self.updateExpression, self.gradDescent_step = build_gradDescent_step( 
								J, Thetabs, 
								alpha=alpha,beta=beta)

	def build_update_with_input(self, X=None, y=None, alpha=0.01, beta=0.0):
		"""
		@type alpha : float
		@param alpha : learning rate
		
		@type beta : float
		@param beta : "momentum" parameter (it's in the update step for gradient descent)

		"""
		
		Thetabs = self._CNN_model.__get_state__()['params']
		if X is None:
			X = self.X
			
		if y is None:
			y = self.y
		
		J = self.J_Theta
		
		self.updateExpression, self.gradDescent_step = build_gradDescent_step( 
								J, Thetabs, X=X,y=y,
								alpha=alpha,beta=beta)


	###############
	# TRAIN MODEL #
	###############	
		
	def train_model_full(self, max_iters=10):
		print "theano.config.allow_gc =: " , theano.config.allow_gc

		train_errors = np.ndarray(max_iters)
		
		learn_grad = self.gradDescent_step
		
		
		for iter in range( max_iters) :
			error=0.
			J_train=learn_grad()
			
			if np.isnan( J_train ) or np.isinf( J_train):
				error += np.array( 1.)
			else: 
				error += np.array( J_train)

			train_errors[iter] = error

		return train_errors

	#################################
	## for Long-Term Serialization ##
	## cf. http://deeplearning.net/software/theano/tutorial/loading_and_saving.html#long-term-serialization
	#################################
	
	def __get_state__(self):
		""" __getstate__(self)  
		
		returns a Python list of numpy arrays, (through theano.shared.get_value() )
		"""
		
		params = self._CNN_model.__get_state__()['params']
		params_vals = [weight.get_value() for weight in params]
		return params_vals
		
	def __set_state__(self, *args):
		""" __setstate__(self,*args)
		
		@type *args : expect a flattened out Python list of Theta* classes
		@param *args : use the "power" of *args so we're flexible about Python function argument inputting (could be a list, could be separate entries)
		"""
		
		self._CNN_model.__set_state__( *args)

			
	def save_parameters(self, filename='objects.save'):
		""" save_parameters (save the weights or parameters of this model as numpy arrays that are pickled)
		
		@type filename : string
		@param filename : string to name the saved file
		"""	
		param_vals = self.__get_state__()
		number_of_params = len(param_vals )
		
		for param_idx in range(number_of_params):
			f = open(filename+str(param_idx),'wb')
			np.save(f, param_vals[param_idx] )
			f.close()
		


	
def build_cost_functional_L2norm(h,y_sym,Thetas):
	""" 
	build_cost_functional_L2norm (with regularization) J=J_y(Theta,b) # J\equiv J_y(\Theta,b), 
	for the L2 norm, or Euclidean space norm
	
	INPUT/PARAMETERS
	================
	@type y_sym  : theano symbolic matrix, such as T.matrix() or theano shared variable
	@param y_sym : output data as a symbolic theano variable or theano shared variable
NOTE: y_sym = T.matrix(); # this could be a vector, but I can keep y to be "general" in size dimensions
	
	@type h     : theano shared variable of size dims. (K,m) (size dim. might be (m,K) due to right action
	@param h    : hypothesis

	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the layers l=1,2,...L-1
	NOTE: remember, we want a list of theano MATRICES, themselves, not the class

	RETURN/OUTPUTS
	==============
	@type J_theta : theano symbolic expression (computational graph)

	"""
	J_theta = np.cast[theano.config.floatX](0.5) * T.mean(T.sqr(h-y_sym))

	return J_theta

def build_cost_functional_L2norm_w_reg(lambda_val,h,y_sym,Thetas):
	""" 
	build_cost_functional_L2norm (with regularization) J=J_y(Theta,b) # J\equiv J_y(\Theta,b), 
	for the L2 norm, or Euclidean space norm, but now with 
	regularization

	INPUT/PARAMETERS
	================
	@type y_sym  : theano symbolic matrix, such as T.matrix() or theano shared variable
	@param y_sym : output data as a symbolic theano variable or theano shared variable
NOTE: y_sym = T.matrix(); # this could be a vector, but I can keep y to be "general" in size dimensions
	
	@type h     : theano shared variable of size dims. (K,m) (size dim. might be (m,K) due to right action
	@param h    : hypothesis

	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the layers l=1,2,...L-1
	NOTE: remember, we want a list of theano MATRICES, themselves, not the class

	RETURN/OUTPUTS
	==============
	@type J_theta : theano symbolic expression (computational graph)

	"""
	J_theta = np.cast[theano.config.floatX](0.5) * T.mean(T.sqr(h-y_sym))

	# T.sqr is element-wise operation (take the square of each element), and so it's an automorphism
	reg_term = T.mean( [ T.sum( T.sqr(Theta), acc_dtype=theano.config.floatX) for Theta in Thetas], acc_dtype=theano.config.floatX )
	reg_term = np.cast[theano.config.floatX](lambda_val/ (2.))*reg_term

	J_theta = J_theta + reg_term
	return J_theta


def build_cost_functional_logistic(h, y, Thetas):
	""" @fn :: build_cost_functional J=J_y(Theta,b) # J\equiv J_y(\Theta,b), but now with 
	X,y being represented as theano symbolic variables first, before the actual numerical data values are given

	INPUT/PARAMETERS
	================	
	@type h     : theano shared variable of size dims. (K,m)
	@param h    : hypothesis
		
	@type y  : theano symbolic matrix, such as T.matrix() 
	@param y : output data as a symbolic theano variable
NOTE: y_sym = T.matrix(); # this could be a vector, but I can keep y to be "general" in size dimensions
	
	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the layers l=1,2,...L-1
	NOTE: remember, we want a list of theano MATRICES, themselves, not the class

	"""
	# logistic regression cost function J, with no regularization (yet)
	J_theta = T.mean( T.sum(
			- y * T.log(h) - (np.float32(1)-y) * T.log( np.float32(1) - h), axis=1))

	return J_theta
	
	
def build_cost_functional_logistic_w_reg(lambda_val, h, y, Thetas):
	""" @fn :: build_cost_functional J=J_y(Theta,b) # J\equiv J_y(\Theta,b), but now with 
	X,y being represented as theano symbolic variables first, before the actual numerical data values are given

	INPUT/PARAMETERS
	================
	@type h     : theano shared variable of size dims. (K,m)
	@param h    : hypothesis
		
	@type y  : theano symbolic matrix, such as T.matrix() 
	@param y : output data as a symbolic theano variable
NOTE: y_sym = T.matrix(); # this could be a vector, but I can keep y to be "general" in size dimensions
	
	@type Thetas : tuple, list, or (ordered) iterable of Theta's as theano shared variables, of length L
	@params Thetas : weights or parameters thetas for all the layers l=1,2,...L-1
	NOTE: remember, we want a list of theano MATRICES, themselves, not the class

	"""
	# logistic regression cost function J, with no regularization (yet)
	J_theta = T.mean( T.sum(
			- y * T.log(h) - (np.float32(1)-y) * T.log( np.float32(1) - h), axis=1))
	
	# T.sqr is element-wise operation (take the square of each element), and so it's an automorphism
	reg_term = T.mean( [ T.sum( T.sqr(Theta), acc_dtype=theano.config.floatX) for Theta in Thetas], acc_dtype=theano.config.floatX )
	reg_term = np.cast[theano.config.floatX](lambda_val/ (2.))*reg_term

	J_theta = J_theta + reg_term
	return J_theta
	

def build_cost_functional_xent(h,y,Thetas):
	"""
	xent - cross entropy
	"""
#	J_binary=T.nnet.categorical_crossentropy(h,y).mean()
#	J_categorical=T.nnet.categorical_crossentropy(h,y)
	J_binary=T.nnet.binary_crossentropy(h,y)

#	return J_categorical
	return J_binary
	

def build_cost_functional_xent_w_reg(lambda_val,h,y,Thetas):
	"""
	xent - cross entropy
	"""
#	J_binary=T.nnet.categorical_crossentropy(h,y).mean()
#	J_categorical=T.nnet.categorical_crossentropy(h,y)
	J_binary=T.nnet.binary_crossentropy(h,y)

	# T.sqr is element-wise operation (take the square of each element), and so it's an automorphism
	reg_term = T.mean( [ T.sum( T.sqr(Theta), acc_dtype=theano.config.floatX) for Theta in Thetas], acc_dtype=theano.config.floatX )
	reg_term = np.cast[theano.config.floatX](lambda_val/ (2.))*reg_term

#	J_categorical += reg_term
	J_binary += reg_term

#	return J_categorical
	return J_binary


def build_gradDescent_step( J, Thetabs, X=None, y=None, alpha =0.01, beta = 0.0):
	""" build_gradDescent_step - gradient Descent (with momentum), but from build_cost_functional for the J

	INPUT/PARAMETERS:
	=================
	@param J     : cost function (from build_cost_function)
	
	@type Thetabs  : Python list (or tuple or iterable) of Thetas, weights matrices, and intercept "row" arrays or theano equivalent
	@param Thetabs : weights or i.e. parameters
	
	
	@param alpha : learning rate
	
	@param beta  : "momentum" constant (parameter)

	RETURN(S)
	=========
	@type updateThetas, gradientDescent_step : tuple of (list of theano symbolic expression, theano function)

	"""
	updateThetabs = [ sandbox.cuda.basic_ops.gpu_from_host( 
					Theta - np.float32( alpha) * T.grad( J, Theta) + np.float32(beta)*Theta ) for Theta in Thetabs]
	

	if X is not None and y is not None:
		gradientDescent_step = theano.function(inputs = [X,y],
											outputs = J, 
											updates = zip(Thetabs,updateThetabs) )
		
	else:
		gradientDescent_step = theano.function(inputs = [],
											outputs = J, 
											updates = zip(Thetabs,updateThetabs) )

	return updateThetabs, gradientDescent_step










