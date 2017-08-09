"""
	@file   : CNN_tf.py
	@brief  : Convolution Neural Network with tensorflow
	@author : Ernest Yeung	ernestyalumni@gmail.com
	@date   : 20170808
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
import tensorflow
import tensorflow as tf
import numpy as np

import DNN_tf
from DNN_tf import Axon

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
	def __init__(self, l, C_ls, Wl, Pl=None, Ll=None,alm1=None, c=None, b=None, activation=None, rngseed=None, padding="VALID" ):
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

		@ref :: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
		@note :: c, in the tensorflow syntax, will be a filter with "shape"  
			[filter_height, filter_width, in_channels, out_channels]
			
			alm1, in the tensorflow syntax, is the input tensor in this case, with "shape" 
			[batch, in_height, in_width, in_channels]
		"""
		C_lm1, C_l = C_ls

		# num input feature maps * filter height * filter width 
		fan_in = C_lm1 * np.prod( Wl)
	
		# num output feature maps * filter height * filter width / pooling size 
		if Pl is not None:
			fan_out = C_l * np.prod(Wl) // np.prod(Pl)
		else:
			fan_out = C_l * np.prod(Wl) 
			
		# make the filter size out of C_ls and Wl
		filter_size = Wl + (C_lm1,C_l)  # [filter_height, filter_width, in_channels, out_channels]
		assert len(filter_size) == (2+len(Wl))

		if c is None:
			minval = -np.sqrt(6. / ( fan_in + fan_out)) 
			maxval = np.sqrt(6. / (fan_in + fan_out))

			if rngseed is None:
				tf.set_random_seed(1234)
				rand_unif_init_vals = tf.random_uniform(filter_size,minval=minval,maxval=maxval)

			else:
				rand_unif_init_vals = tf.random_uniform(filter_size,minval=minval,maxval=maxval,seed=rngseed)

			if activation == tf.sigmoid:
				rand_unif_init_vals *= np.float32( 4 )


			c = tf.Variable(rand_unif_init_vals, name="cl"+str(l) ,dtype=tf.float32)  
			
		# store Axon's bias
		if b is None:
			b = tf.Variable(tf.random_normal([C_l,]) , dtype=tf.float32 )
	
		# input tensor of shape [batch, in_height, in_width, in_channels]  
		# cf. https://www.tensorflow.org/programmers_guide/faq
		# f you use placeholders for feeding input, you can specify a variable batch dimension by creating the placeholder 
		# with tf.placeholder(..., shape=[None, ...]). The None element of the shape corresponds to a variable-sized dimension.
		if alm1 is None:
			if Ll is None:
				alm1 = tf.placeholder(tf.float32, shape=[None,None,None,C_lm1], name="alm1"+str(l))
			else:
				image_shape = [None,] + list(Ll) + [C_lm1,] 
				alm1 = tf.placeholder(tf.float32, shape=image_shape, name="alm1"+str(l))
				
		self.c = c  # size dims. (W_1,...W_d,C_lm1,C_l) i.e. W_1,x ... x W_d x C_lm1 x C_l    
		self.b     = b      # dims. C_l
		self.alm1    = alm1     # dims. (m,L_1,...L_d,C_lm1)
		self.C_ls = C_ls
		self.Wl = Wl
		self.Pl = Pl
		
		self.l     = l

		if activation is None:
			self.psi = None
		else:
			self.psi = activation

		# do convolution, do a connect through
		conv_out=tf.nn.conv2d( 
				self.alm1, 
				self.c, 
				strides=[1,1,1,1], 
				padding=padding, 
				use_cudnn_on_gpu=True,
				name=None)

		if self.psi is None:
			zl = conv_out + self.b
		else:
			zl = self.psi( conv_out + self.b)
			
				
		# cf. https://www.tensorflow.org/api_docs/python/tf/nn/max_pool  
		# input - A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
		# ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
		# strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
		
		if Pl is not None:
			try:
				window_size = (1,) + Pl + (1,)
				window_size = list(window_size) 
			except TypeError: # consider a list was inputted
				window_size = [1,] + Pl + [1,]

			pooled_out = tf.nn.max_pool(zl,
										ksize=window_size,
										strides=window_size, 
										padding=padding)
			self.al = pooled_out
			
		else:
			self.al = zl


	def connect_through(self, alm1_in=None, Pl=None, padding="VALID"):
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
		filter_shape = Wl + (C_lm1,C_l)
		assert len(filter_shape) == (2+len(Wl))

		# convolve input feature maps with filters
		conv_out=tf.nn.conv2d( 
								alm1, 
								c, 
								strides=[1,1,1,1], 
								padding=padding, 
								use_cudnn_on_gpu=True, name=None)

		if self.psi is None:
			zl = conv_out + self.b
		else:
			zl = self.psi( conv_out + self.b)


		if Pl is not None:
			try:
				window_size = (1,) + Pl + (1,)
				window_size = list(window_size) 
			except TypeError: # consider a list was inputted
				window_size = [1,] + Pl + [1,]

			pooled_out = tf.nn.max_pool(zl,
										ksize=window_size,
										strides=window_size, 
										padding=padding)
			self.al = pooled_out
			
		else:
			self.al = zl

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
	def __init__(self, L, CorD, dim_data, a0=None,activation_fxn=tf.tanh, psi_L=tf.nn.sigmoid, rngseed=None,padding="VALID" ):
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

			if a0 is None:
				image_shape = [None,] + list(L1) + [C_0,] 
				a0 = tf.placeholder(tf.float32, shape=image_shape,name="al0")

			Axon1=Axon_CNN(1, (C_0,C_1), W1,Pl=P1,Ll=L1,alm1=a0,activation=activation_fxn, rngseed=rngseed,padding=padding)
	
			Axon1.connect_through(Pl=P1,padding=padding)
			Axons_lst.append( Axon1)
		elif CorD[0] is "D":
			s_l= dim_data[0]
			if a0 is None:
				input_shape = [None, s_l[0]]
				a0 = tf.placeholder(tf.float32, shape=input_shape,name="al0")

			Axon1 = Axon(1, (s_l[0],s_l[1]), alm1=a0, activation=activation_fxn, rngseed=rngseed)
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
								if padding is "VALID":
									i_sizedim_after_pooling = (Llm1[i]-Wlm1[i]+1) // Plm1[i]
									try:
										assert(i_sizedim_after_pooling == Ll[i] )
									except AssertionError as Asserterr:
										print(Asserterr,
											" for %d th dim., for Axon %d, we obtain incorrect size dim. after pooling. " % i, l)
							else:
								if padding is "VALID":
									#i_sizedim_after_filter = (size_dim_Llm1[i]-Wlm1[i]+1) 
									i_sizedim_after_filter = (Llm1[i]-Wlm1[i]+1) 
									try:
										assert(i_sizedim_after_filter == Ll[i] )
									except AssertionError as Asserterr:
										print(Asserterr,
											" for %d th dim., for Axon %d, we obtain incorrect size dim. after filtering. " % i, l)
							
				# END of sanity dimensions check

				Axonl=Axon_CNN(l, (C_lm1,C_l), Wl, Pl=Pl, Ll=Ll, alm1=inputlayer_alm1,
									activation=activation_fxn, rngseed=rngseed)
				Axonl.connect_through(Pl=Pl,padding=padding)
				Axons_lst.append( Axonl)
			elif CorD[l-1] is "D":
				# for the case of when we go from a convolution axon to a DNN axon, we have to flatten the input
				s_l= dim_data[l-1]


				if CorD[l-2] is "C": 
					# get the flattened "shape" or size-dimensions  
					"""
					input_shape = len(inputlayer_alm1.shape) # should be of size dims. m x (L1...Ld) x C_lm1
					flattened_sizedim = inputlayer_alm1.shape[1]
					for idx in range(2,input_shape+1):
						flattened_sizedim *= inputlayer_alm1.shape[idx]
					
					flattened_sizedim = int(flattened_sizedim)
					
					inputlayer_alm1 = tf.reshape( inputlayer_alm1, shape[None, flattened_sizedim ] )

#					inputlayer_alm1 = tf.reshape( inputlayer_alm1, shape=[None,-1] ) # or try this, EY : 20170808
					"""	
#					try:
#						inputlayer_alm1 = tf.reshape( inputlayer_alm1, shape=[None, int(s_l[0])] )
#					except TypeError as typeerr:
#						print( typeerr, " l : ",  l )
					inputlayer_alm1 = tf.reshape( inputlayer_alm1, shape=[ -1, int(s_l[0])] )
						

				Axonl = Axon(l, (s_l[0],s_l[1]), alm1=inputlayer_alm1,activation=activation_fxn, rngseed=rngseed)
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

				AxonL=Axon_CNN(L, (C_Lm1,C_L), WL,Pl=PL, Ll=LL,alm1=inputlayer_aLm1,
									activation=psi_L, rngseed=rngseed,padding=padding)
				AxonL.connect_through(Pl=PL,padding=padding)
				Axons_lst.append( AxonL)
			elif CorD[L-1] is "D":
				# for the case of when we go from a convolution axon to a DNN axon, we have to flatten the input
				s_l = dim_data[L-1]
				
				if CorD[L-2] is "C": 
#					inputlayer_aLm1 = inputlayer_aLm1.flatten(2)
#					try:
#					except TypeError:
#						inputlayer_aLm1 = tf.reshape( inputlayer_aLm1, shape=list([None, s_l[0]]) )
					inputlayer_aLm1 = tf.reshape( inputlayer_aLm1, shape=[-1, int(s_l[0])] )

	
				AxonL = Axon(L,(s_l[0],s_l[1]),alm1=inputlayer_aLm1, activation=psi_L, rngseed=rngseed)

				AxonL.connect_through()
					
				Axons_lst.append(AxonL)
		
		self.Axons = Axons_lst
		

	def connect_through(self, X_in=None,padding="VALID"):
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

			self.Axons[0].connect_through(Pl=P1,padding=padding)
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

				self.Axons[idx].connect_through(Pl=Pl,padding=padding)
			elif CorD[idx] is "D":
				s_l= dim_data[idx]

				if CorD[idx-1] is "C":
#					self.Axons[idx].alm1 = tf.reshape( self.Axons[idx-1].al , shape=[None, s_l[0]] )
					self.Axons[idx].alm1 = tf.reshape( self.Axons[idx-1].al , shape=[-1, int(s_l[0]) ] )

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
	def __init__(self,CNN_model,X=None,y=None,padding="VALID"):
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
		self.L = CNN_model.L
		L = self.L

		CorD = CNN_model._CorD
		dim_data = CNN_model._dim_data

		if X is None:
			if CorD[0] is "C":
				C_0,C_1=dim_data[0]['C_ls']
				W1 = dim_data[0]['Wl']
				P1 = dim_data[0]['Pl']
				L1=dim_data[0]['Ll']

				image_shape = [None,] + list(L1) + [C_0,] 
				
				self.X = tf.placeholder(tf.float32, shape=image_shape, name="X")

			elif CorD[0] is "D":
				s_l= dim_data[0]
				input_shape = [None, s_l[0]]
				self.X = tf.placeholder(tf.float32, shape=input_shape,name="X")

		else:
			self.X = X

		if y is None:
			if CorD[L-1] is "C":
				C_Lm1,C_L=dim_data[L-1]['C_ls']
				WL = dim_data[L-1]['Wl']
				PL = dim_data[L-1]['Pl']
				LL=dim_data[L-1]['Ll']

				output_shape = []

				if LL is not None: 
					d = len(LL)
					for i in range(d):  # i=0,1,...d-1 
						if PL is not None:
							if padding is "VALID":
								sizedim_after_pooling_L = (LL[i]-WL[i]+1) // PL[i]
								output_shape.append( sizedim_after_pooling_L)
							else:
								output_shape.append( LL[i] )

						else:
							if padding is "VALID":
								sizedim_after_filter_L = (LL[i]-WL[i]+1) 
								output_shape.append( sizedim_after_pooling_L)
							else:
								output_shape.append( LL[i] )
					output_shape = [None,] + output_shape + [C_L,] 
				else:
					output_shape = [None,] + [LL,] + [C_L,]	
				
				self.y=tf.placeholder(tf.float32, shape=output_shape,name="y")
				

			elif CorD[L-1] is "D":
				s_l= dim_data[L-1]
				output_shape = [None, s_l[1]]
				self.y = tf.placeholder(tf.float32, shape=output_shape,name="y")

		else:
			self.y = y

	def connect_through(self,X=None,padding="VALID"):
		""" connect_through - 
		
		@brief :: connect through the Axons (the actual feedforward operation)

		"""
		if X is not None:
			self.X = X
		else: 
			X = self.X
		
		h = self._CNN_model.connect_through(X,padding=padding)
		self.yhat = self._CNN_model._get_outer_layer_()
		return h 

	###############################
	# BUILD cost functions J ######
	###############################

	def build_J_L2norm(self,y=None):
		""" build_J_L2norm

		J = \frac{1}{m} \sum_{i=0}^{m-1} (\widehat{\mathbf{y}}^{(i)}-\mathbf{y}^{(i)})^2
		where ^2 implies that the Euclidean metric is used, as in general y\in \mathbb{K}^K, where \mathbb{K} is some field
		
		"""
		if y is not None:
			self.y = y
		else:
			y = self.y
		
		# get the outer layer of the network, which is the hypothesis
		yhat = self.yhat
		
		# tf.nn.l2_loss - computes half the L2 norm of a tensor without the sqrt, l2_loss(t, name=None)
		J = 0.5 * tf.reduce_mean( 
				tf.reduce_sum( tf.square(yhat-y), # elements-wise square 
								axis=1, name="squared_error"), # Euclidean metric
									name="mean_squared_error") # average over all m (input) samples
		self.J_Theta = J
		return J 

	def build_J_L2norm_w_reg(self,lambda_val=1.0,y=None):
		""" build_J_L2norm_w_reg
		
		with 
		J_loss = \frac{1}{m} \sum_{i=0}^{m-1} (\widehat{\mathbf{y}}^{(i)}-\mathbf{y}^{(i)})^2
		(see build_J_L2norm)
		this adds the following term:  
		J_reg = \sum_{l=0}^{L-1} \| \Theta^{(l)} \|_2 = 
				= \sum_{l=0}^{L-1} \sum_{I \in \mathcal{I}} (\Theta_I^{(l)})^2 
		and computes 
		J = J_loss + J_reg
		
		Notice that in J_reg, there's no $\frac{1}{m}$ factor, m=number of (input) examples/samples, and there is no way to 
		obtain that factor without invoking m haphazardly from the matrix size dimension of the input X since 
		X \in \text{Mat}_{\mathbb{K}}(m,d).  This is done in build_J_L2norm_w_scaledreg
		
		@type lambda_val  : (single) float number
		@param lambda_val : regularization parameter
		
		"""
		"""		
		if y is not None:
			self.y = y
		else:
			y = self.y
		"""
		loss = self.build_J_L2norm(y)
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		
		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n( ThetaL2norms )
		
		J = loss + lambda_val*reg_term  

		self.J_Theta = J

		return J 

	def build_J_L2norm_w_scaled_reg(self,lambda_val=1.0,y=None):
		""" build_J_L2norm_w_scaled_reg
		
		with 
		J_loss = \frac{1}{m} \sum_{i=0}^{m-1} (\widehat{\mathbf{y}}^{(i)}-\mathbf{y}^{(i)})^2
		(see build_J_L2norm)
		this adds the following term:  
		J_reg = \frac{1}{m} \sum_{l=0}^{L-1} \| \Theta^{(l)} \|_2 = 
				= \frac{1}{m} \sum_{l=0}^{L-1} \sum_{I \in \mathcal{I}} (\Theta_I^{(l)})^2 
		and computes 
		J = J_loss + J_reg
				
		@type lambda_val  : (single) float number
		@param lambda_val : regularization parameter
		
		"""
		"""		
		if y is not None:
			self.y = y
		else:
			y = self.y
		"""
		# m=number of samples, i.e. total "batch" size
#		m = tf.cast( self.X.get_shape()[0], tf.float32) # ValueError
		try:
			m = tf.cast( self.X.get_shape()[0], tf.float32) # ValueError
		except ValueError as valerr:
			print("ValueError in obtaining batch size: ", valerr)
			m = self.X.get_shape()[0]
			
		
		loss = self.build_J_L2norm(y)
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		
		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n( ThetaL2norms )
		
		J = loss + lambda_val*(1.0/m)*reg_term  

		self.J_Theta = J

		return J 		

	def build_J_logistic(self, y=None):
		""" @fn :: build_J_logistic - 
			@brief :: build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)
		"""
		if y is not None:
			self.y = y
		else:
			y = self.y

		# get the outer layer of the network, which is the hypothesis
		yhat = self.yhat

#		Thetas_only = self._CNN_model.__get_state__()['Thetas']

		J = tf.reduce_mean( tf.reduce_sum(-y*tf.log(yhat)-(1.-y)*tf.log(1. -yhat),axis=1) )
		
		self.J_Theta = J
		return J

	def build_J_logistic_w_reg(self, lambda_val=1.0, y=None ):
		""" @fn :: build_J_L2norm_w_reg - 
		@brief :: build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)

		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		"""
		"""		
		if y is not None:
			self.y = y
		else:
			y = self.y
		"""		
		loss = self.build_J_logistic(y)

		Thetas_only = self._CNN_model.__get_state__()['Thetas']

		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n(ThetaL2norms)
		
		J = loss + lambda_val*reg_term

		self.J_Theta = J
		return J

	def build_J_logistic_w_scaled_reg(self, lambda_val=1.0 ,y=None ):
		""" @fn :: build_J_L2norm_w_reg - 
		@brief :: build or make cost functional, of the form of the L2 norm (i.e. Euclidean distance norm)

		# regularization, "learning", "momentum" parameters/constants/rates
		@type lambda_val : float
		@param lambda_val : regularization constant
		"""

		try:
			m = tf.cast( self.X.get_shape()[0], tf.float32) # ValueError
		except ValueError as valerr:
			print("ValueError in obtaining batch size: ", valerr)
			m = self.X.get_shape()[0]

		loss = self.build_J_logistic(y)

		Thetas_only = self._CNN_model.__get_state__()['Thetas']

		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n(ThetaL2norms)
		
		J = loss + lambda_val*(1.0/m)*reg_term

		self.J_Theta = J
		return J
		
	def build_J_xent(self,y=None):
		""" build_J_xent
		This makes J for the multidimensional (for y \in \mathbb{K}^K)
		
		"""	
		if y is not None:
			self.y = y
		else:
			y = self.y
		
		# get the outer layer of the network, which is the hypothesis
		yhat = self.yhat
		
		J = tf.reduce_mean(
				tf.reduce_sum( 
					tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=yhat), # elements-wise logistic (entropy) function 
						axis=1, name="logistic_sigmoid_error"), # summation=k=0,1,...K-1 over dimensions of y \in \mathbb{K}
							name="mean_error") # average over all m (input) samples
					 	
		self.J_Theta = J
		return J 	

	def build_J_xent_w_reg(self, lambda_val=1.0, y=None):
		""" build_J_xent_w_reg
		
		with 
		J_loss 
		(see build_J_xent)
		this adds the following term:  
		J_reg = \sum_{l=0}^{L-1} \| \Theta^{(l)} \|_2 = 
				= \sum_{l=0}^{L-1} \sum_{I \in \mathcal{I}} (\Theta_I^{(l)})^2 
		and computes 
		J = J_loss + J_reg
		
		Notice that in J_reg, there's no $\frac{1}{m}$ factor, m=number of (input) examples/samples, and there is no way to 
		obtain that factor without invoking m haphazardly from the matrix size dimension of the input X since 
		X \in \text{Mat}_{\mathbb{K}}(m,d).  This is done in build_J_L2norm_w_scaledreg
		
		@type lambda_val  : (single) float number
		@param lambda_val : regularization parameter
		
		"""
		"""		
		if y is not None:
			self.y = y
		else:
			y = self.y
		"""
		loss = self.build_J_xent(y)
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		
		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n( ThetaL2norms )
		
		J = loss + lambda_val*reg_term  

		self.J_Theta = J

		return J 
		
	def build_J_xent_w_scaled_reg(self, lambda_val=1.0, y=None):
		""" build_J_L2norm_w_scaled_reg
		
		with 
		J_loss = \frac{1}{m} \sum_{i=0}^{m-1} (\widehat{\mathbf{y}}^{(i)}-\mathbf{y}^{(i)})^2
		(see build_J_L2norm)
		this adds the following term:  
		J_reg = \frac{1}{m} \sum_{l=0}^{L-1} \| \Theta^{(l)} \|_2 = 
				= \frac{1}{m} \sum_{l=0}^{L-1} \sum_{I \in \mathcal{I}} (\Theta_I^{(l)})^2 
		and computes 
		J = J_loss + J_reg
				
		@type lambda_val  : (single) float number
		@param lambda_val : regularization parameter
		
		"""
		"""		
		if y is not None:
			self.y = y
		else:
			y = self.y
		"""
		# m=number of samples, i.e. total "batch" size
		X=self.X

		try:
			m = tf.cast( self.X.get_shape()[0], tf.float32) # ValueError
		except ValueError as valerr:
			print("ValueError in obtaining batch size: ", valerr)
			m = self.X.get_shape()[0]

		
		loss = self.build_J_xent(y)
		
		Thetas_only = self._CNN_model.__get_state__()['Thetas']
		
		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n( ThetaL2norms )
		
		J = loss + lambda_val*(1.0/m)*reg_term  

		self.J_Theta = J

		return J 		
	
	###############################
	# BUILD MODEL'S OPTIMIZER 	  #
	###############################

	def build_optimizer(self, alpha=0.01):
		""" build_optimizer
		
		@type alpha  :
		@param alpha : learning rate
		"""
		J = self.J_Theta

		# Note, minimize() knows to modify W i.e. Theta and b because Variable objects are trainable=True by default
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(J)
	
		self._optimizer = optimizer

		return optimizer

	def build_optimizer_Adagrad(self, alpha=0.01):
		""" build_optimizer_Adagrad
		
		@brief : build optimizer that uses Adagrad
		
		@type alpha  :
		@param alpha : learning rate
		"""
		J = self.J_Theta

		# Note, minimize() knows to modify W i.e. Theta and b because Variable objects are trainable=True by default
		optimizer = tf.train.AdagradOptimizer(learning_rate=alpha).minimize(J)
	
		self._optimizer = optimizer
	
		return optimizer  

	def build_optimizer_Adam(self, alpha=1e-4):
		""" @fn build_optimizer_Adam 
		
		@brief : build optimizer that uses Adam
		
		@type alpha  :
		@param alpha : learning rate
		"""
		J = self.J_Theta
		
		optimizer = tf.train.AdamOptimizer(alpha).minimize(J)
		
		self._optimizer = optimizer
		
		return optimizer


	###############
	# TRAIN MODEL #
	###############	

	def train_model(self, max_iters=10, X_data=None,y_data=None,session_configuration=None):
		""" train_model
		
		@type   session_configuration : 
		@params session_configuration : example: tf.ConfigProto(log_device_placement=True) 
		
		@ref : https://www.tensorflow.org/tutorials/using_gpu  
		
		Allowing GPU memory growth 
		"In some cases it is desirable for the process to only allocate a subset of the available memory, 
		or to only grow the memory usage as is needed by the process. 
		TensorFlow provides two Config options on the Session to control this.

		The first is the allow_growth option, 
		which attempts to allocate only as much GPU memory based on runtime allocations: 
		it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, 
		we extend the GPU memory region needed by the TensorFlow process. 
		Note that we do not release memory, since that can lead to even worse memory fragmentation. 
		To turn this option on, set the option in the ConfigProto by:"

		session_configuration = tf.ConfigProto()
		session_configuration.gpu_options.allow_growth=True  
		
		"The second method is the per_process_gpu_memory_fraction option, 
		which determines the fraction of the overall amount of memory that each visible GPU should be allocated. 
		For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU by:"
		
		session_configuration = tf.ConfigProto()  
		session_configuration.gpu_options.per_process_gpu_memory_fraction =0.4  
		
		"""
		if X_data is not None:
			# cf. https://stackoverflow.com/questions/37596333/tensorflow-store-training-data-on-gpu-memory
			# but TypeError obtained: he value of a feed cannot be a tf.Tensor object. 
			# Acceptable feed values include Python scalars, strings, lists, numpy ndarrays, or TensorHandles.
#			with tf.device('/gpu:0'):
				self.X_data = X_data
#				X_data = self.X_data
		else:
			X_data=self.X_data

		if y_data is not None:
#			with tf.device('/gpu:0'):
#				self.y_data = tf.constant(y_data)
#				y_data = self.y_data
				self.y_data = y_data

		else:
			y_data=self.y_data

		train_errors = np.ndarray(max_iters)
		
		optimizer = self._optimizer
		J = self.J_Theta
		X=self.X
		y=self.y

		try:
			sess=self._sess
			print("Found self._sess: \n")
		except AttributeError:
			print("No self._sess: creating self._sess=tf.Session() now: \n")
			if session_configuration is None:
				sess = tf.Session()
			else:
				sess = tf.Session(config=session_configuration)
			self._sess=sess


		# this is needed in the future for saving this model
		# cf. Saving Variables in https://www.tensorflow.org/programmers_guide/variables
		try:
			saver=self._saver
			print("Found self._saver: \n")
		except AttributeError:
			print("No self._saver: creating self._saver=tf.train.Saver() now: \n")
			saver=tf.train.Saver()
			self._saver=saver

		try:
			init_op=self._init_op
			sess.run(init_op)
			print("Found self._init_op, executing sess.run(init_op) now: \n")
		except AttributeError:
			print("No self.__init: creating self.__init=tf.global_variables_initializer() now:")

			init_op=tf.global_variables_initializer()
			self._init_op = init_op
			sess.run(init_op)
		
		
		# "boilerplate" parameters for making the print out look nice
		step_interval = max_iters/10
		
		for iter in range(max_iters): 
			_,err=sess.run([optimizer,J],feed_dict={X:X_data,y:y_data})
			
			train_errors[iter] = err
			if not (iter % step_interval) or iter < 10: 
				print("Iter : %3d, J,cost,i.e. error-train: %4.4f " % (iter,err))
				
		errFinal = train_errors[-1]
		print("Final error: %.19f" % (errFinal) )
		
		return train_errors
	
	
	def build_sess_w_init(self,session_configuration=None):
		if session_configuration is None:
			sess = tf.Session()
		else:
			sess = tf.Session(config=session_configuration)

		init_op=tf.global_variables_initializer()
		sess.run(init_op)
		self._sess = sess
		self._init_op = init_op
		return sess, init_op 

	def build_sess_w_saver(self,session_configuration=None):
		""" build_sess_w_saver 
		@brief : build_sess_w_saver instantiates, or creates a tf.Session and a 
					tf.train.Saver
					which is to aid in restoring a previously saved model  
		"""
		if session_configuration is None:
			sess = tf.Session()
		else:
			sess = tf.Session(config=session_configuration)

		self._sess = sess
	
		saver=tf.train.Saver()

		self._saver = saver
		return sess, saver 

		
	def predict(self,X_data=None):

		if X_data is not None:
			self.X_data = X_data
		else:
			X_data=self.X_data
		
		X    = self.X
		yhat = self.yhat
#		sess = self._sess
		try:
			sess=self._sess
			print("Found self._sess: \n")
		except AttributeError:
			print("No self._sess: creating self._sess=tf.Session() now: \n")
			sess = tf.Session()
			init_op=tf.global_variables_initializer()
			sess.run(init_op)
	
			self._sess=sess

		yhat_val = sess.run(yhat,feed_dict={X:X_data})
		return yhat_val  
		
	def accuracy_score(self,X_data=None, y_data=None):

		if X_data is None:
			X_data=self.X_data
		if y_data is None:
			y_data=self.y_data

		try:
			sess=self._sess
			print("Found self._sess: \n")
		except AttributeError:
			print("No self._sess: creating self._sess=tf.Session() now: \n")
			sess = tf.Session()
			init_op=tf.global_variables_initializer()
			sess.run(init_op)
	
			self._sess=sess
		L=self.L

		X=self.X
		y=self.y
		yhat=self.yhat

		accuracy_score = tf.metrics.accuracy(labels=y,predictions=yhat)
		acc_score=sess.run(accuracy_score,feed_dict={X:X_data,y:y_data})

		return acc_score

#####################
##### File I/O ######		
#####################

	def save_sess(self,filename):
		""" save_sess
		
		"""
		try:
			# sanity check as specified by the API, as all variables have to be initialized 
			self._init_op

			sess=self._sess
			saver = self._saver
			save_path=saver.save(sess, filename)
			print("Model saved in file: %s" % save_path) 
		except AttributeError:
			print "No session to save!"  
			return -1	

	def restore_sess(self,filename):
		""" restore_sess
		
		"""
		try:
			saver=self._saver
			print("Found self._sess: \n")
		except AttributeError:
			saver=tf.train.Saver()
			self._saver=saver
			print("No self._saver: creating self._saver=tf.train.Saver() now: \n")

		try:
			sess=self._sess
			print("Found self._saver: \n")
		except AttributeError:
			sess = tf.Session()
			self._sess=sess
			print("No self._sess: creating self._sess=tf.Session() now: \n")


		


		
		
		
if __name__ == "__main__":
	print("In main")






