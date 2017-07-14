"""
	@file   : DNN_tf.py
	@brief  : Deep Neural Network with tensorFlow (tf)
	@author : Ernest Yeung	ernestyalumni@gmail.com
	@date   : 20170707
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

########################################################################
### Axon class #########################################################
########################################################################

class Axon(object):
	""" Axon - Axon class, for parameters or weights Theta and intercepts (bias) b, 
		between layers l-1 and l, for right action, 
		i.e. the matrix (action) multiplies from the right onto the vector 
		(the module or i.e. vector space)
	
	(AVAILABLE) CLASS MEMBERS
	=========================
	 - Available at initialization __init__
	@type  Theta
	@param Theta

	@type  b
	@param b

	@type  alm1
	@param alm1
	
	@type  psi
	@param psi

	@type  l
	@param l

	
	"""  

	def __init__(self, l, s_ls, alm1=None, Theta=None, b=None, activation=None , rngseed=None): 
		""" Initialize the parameters for the 'Axon'
		
		@type s_ls  : tuple of (positive) integers of size (length) 2, only
		@param s_ls : "matrix" size dimensions of Theta or weight matrix, of 
						size dims. (s_lm1, s_l) (this is important)
						Bottom line: s_ls = (s_lm1,s_l)

		@type l    : (positive) integer
		@param l   : number labeling the Axon in the sequence of Axons - 
						l=1,2,..L ...
						Note that the output layer of the Axon is the lth layer in the network
		
		@type alm1     : Matrix of size dims. (m, s_l), m=1,2..., number of training examples
		@oaram alm1    : "nodes" or "units" of layer (l-1)

		@type Theta  : Matrix of size dims. (s_{l-1},s_l), i.e. matrix dims. (s_{l-1}x(s_l)), i.e. \Theta \in \text{Mat}_{\mathbb{R}}(s_{l-1},s_l) 
		@param Theta : "weights" or parameters for l, l=1,2, ... L
		
		@type b      : vector of dim. s_{l} or s_l; it's now a "row" array of that length s_l
		@param b     : intercepts or, i.e., bias

		@type activation  : function
		@param activation : Non linearity to be applied in the layer(s), specifically to layer l
		
		
		"""
		assert (l>0)
		lstr = str(l)
		s_lm1, s_l = s_ls

		if alm1 is None:
			alm1=tf.placeholder(tf.float32, shape=[None,s_lm1], name="al"+lstr)
		
		# Store Axon's weight
		if Theta is None: 
			minval = -np.sqrt(6. / (s_lm1 + s_l) )
			maxval = np.sqrt(6. / (s_lm1 + s_l) )

			if rngseed is None:
				tf.set_random_seed(1234)
				rand_unif_init_vals = tf.random_uniform([s_lm1, s_l],minval=minval,maxval=maxval)
			else:
				rand_unif_init_vals = tf.random_uniform([s_lm1, s_l],minval=minval,maxval=maxval,seed=rngseed)
			Theta = tf.Variable(rand_unif_init_vals, name="Theta_l"+lstr, dtype=tf.float32)

		# Store Axon's bias
		if b is None:
			# assume bias b is for l>1, i.e. not for an input layer going into layer l=1
			b = tf.Variable(tf.random_normal([s_l,]))
			
		self.Theta = Theta
		self.b     = b
		self.alm1 = alm1 

		self.l = l
		
		self.psi = activation

		
	def connect_through(self, al_in=None):
		""" connect_through  
		
			Note that I made connect_through a separate class method, separate from the automatic initialization, 
			because you can then make changes to the "layer units" or "nodes" before "connecting the layers"
		"""
		if al_in is not None:
			self.alm1 = al_in
			
		alm1  = self.alm1	
		Theta = self.Theta	
		b     = self.b	
			
#		lin_zl = tf.add( tf.matmul(alm1, Theta) , b)
		lin_zl = tf.nn.bias_add( tf.matmul(alm1, Theta) , b)

		if self.psi is None:
			self.al = lin_zl
		else:
			self.al = self.psi( lin_zl)
		return self.al 

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
		self.Theta.assign(Theta_in)
		self.b.assign(b_in)

########################################################################
### END of Axon class ##################################################
########################################################################

########################################################################
## Feedforward (right) class ###########################################
########################################################################

class FeedForward(object): 
	""" Feedforward - Feedforward  
	"""
	def __init__(self, L, s_l, a0=None, activation_fxn=tf.tanh, psi_L=tf.nn.sigmoid, rngseed=None):
		""" Initialize MLP class

		INPUT/PARAMETER(S)
		==================
		@type L : (positive) integer
		@param L : total number of axons L, counting l=0 (input layer), and l=L (output layer), so only 1 hidden layer means L=2

		@type s_l : Python list (or tuple, or iterable) of (positive) integers
		@param s_l : list (or tuple or iterable) of length L+1, 
					containing 
						(positive) integers for s_l, size or number of "nodes" or "units" in layer l=0,1,...L; 
					NOTE that number of "components" of y, K, must be equal to s_L, s_L=K
		"""
		self.L = L
		self.s_l = s_l
		
		# sanity check (number of dimensions for the layers and Axon is same as for L, number of axons, but L+1
		assert( len(s_l) == (L+1))
		
		###############
		# BUILD MODEL #
		###############		
		Axons_lst = [] # temporary list of Axons
		
		if a0 is None:
			a0=tf.placeholder(tf.float32, shape=[None,s_l[0]],name="al0")

		# initialize an instance of class Axon 
		Axon1 = Axon(1,(s_l[0],s_l[1]),alm1=a0, activation=activation_fxn,rngseed=rngseed)
		Axon1.connect_through()
		Axons_lst.append(Axon1)

		for l in range(2,L): # don't include the output layer yet  
			Axonlm1 = Axons_lst[-1]
			inputlayer_alm1 = Axonlm1.al
			
			# initialize an instance of class Axon
			Axonl = Axon(l,(s_l[l-1],s_l[l]),alm1=inputlayer_alm1, activation=activation_fxn,rngseed=rngseed)
			Axonl.connect_through()
			Axons_lst.append(Axonl)
			
		# output layer
		if (L>1):
			Axonlm1=Axons_lst[-1]
			inputlayer_alm1 = Axonlm1.al
			
			# initialize an instance of Axon
			Axonl = Axon(L,(s_l[L-1],s_l[L]),alm1=inputlayer_alm1,activation=psi_L,rngseed=rngseed)
			Axonl.connect_through()
			Axons_lst.append(Axonl)

		self.Axons = Axons_lst

		
	def connect_through(self, X_in=None):
		""" connect_through - connect through the layers (the actual feedforward operation)

		INPUTS/PARAMETERS
		=================
		@type X_in : variable but with values set

		"""
		if X_in is not None:
			self.Axons[0].alm1 = X_in
		
		self.Axons[0].connect_through()
		
		L = self.L
		
		for l in range(1,L):  	# l=1,2,...L-1, for each of the Axons 
			self.Axons[l].alm1 = self.Axons[l-1].al
			self.Axons[l].connect_through()	
		
		# return the result of Feedforward operation, denoted h_Theta
		h_Theta = self.Axons[-1].al
		return h_Theta

	def __get_state__(self):
		""" __get_state__ - return the parameters or "weights" that were used in this feedforward
		
		"""
		Axons = self.Axons

		Thetas = [theta for Axon in Axons for theta in Axon.__get_state__()['Thetas'] ]
		bs     = [b for Axon in Axons for b in Axon.__get_state__()['bs'] ]
		params = [weight for Axon in Axons for weight in Axon.__get_state__()['params']]

		return dict(Thetas=Thetas,bs=bs,params=params)
		
		
	def _get_outer_layer_(self):
		h_Theta = self.Axons[-1].al
		return h_Theta


class DNN(object):
	""" DNN - Deep Neural Network 
	
	(AVAILABLE) CLASS MEMBERS
	=========================
	
	"""
	def __init__(self, DNN_model,X=None,y=None):
		""" Initialize DNN class 
		
		INPUTS/PARAMETERS
		=================
		@type DNN_model  : instance of FeedForward class
		@param DNN_model : instance of FeedForward class

		@type x  : 
		@param x : x represents input (sample) data, usually of matrix size dimensions (m,d), with m=number of samples, d=number of features

		@type y  :
		@param y : y represents target data, usually of matrix size dimension (m,K)

		"""
		
		self.DNN_model = DNN_model  
		self.L = DNN_model.L
		L = self.L
		Theta0 = DNN_model.Axons[0].Theta # l=1 for 1-based counting, l=0 for 0-based counting
		d = Theta0.get_shape()[0] # d=number of features
		ThetaL = DNN_model.Axons[L-1].Theta
		K = ThetaL.get_shape()[-1] # K=dim. of target data space 
		
		if X is None:
			self.X = tf.placeholder(tf.float32, shape=(None,d), name="X")
		else:
			self.X = X
			
		if y is None:
			self.y = tf.placeholder(tf.float32,shape=(None,K), name='y')
		else:
			self.y = y

		self.DNN_model.connect_through(self.X)

		self.yhat = self.DNN_model._get_outer_layer_()

	def connect_through(self, X=None):
		if X is not None:
			self.X = X
		else:
			X = self.X

		if y is not None:
			self.y = y 
		else:
			y = self.y 
		
		self.DNN_model.connect_through(X_in=X)
		self.yhat = self.DNN_model._get_outer_layer_()
		return self.yhat
		
			
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
			
	def build_J_L2norm_w_reg(self,y=None,lambda_val=1.0):
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
		loss = self.build_J_L2norm(y)
		
		Thetas_only = self.DNN_model.__get_state__()['Thetas']
		
		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n( ThetaL2norms )
		
		J = loss + lambda_val*reg_term  

		self.J_Theta = J

		return J 
		
	def build_J_L2norm_w_scaled_reg(self,y=None,lambda_val=1.0):
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
		# m=number of samples, i.e. total "batch" size
		m = tf.cast( self.X.get_shape()[0], tf.float32)
		
		loss = self.build_J_L2norm(y)
		
		Thetas_only = self.DNN_model.__get_state__()['Thetas']
		
		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n( ThetaL2norms )
		
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
	
	def build_J_xent_w_reg(self,y=None,lambda_val=1.0):
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
		loss = self.build_J_xent(y)
		
		Thetas_only = self.DNN_model.__get_state__()['Thetas']
		
		ThetaL2norms = map( tf.nn.l2_loss, Thetas_only)
		
		reg_term = tf.accumulate_n( ThetaL2norms )
		
		J = loss + lambda_val*reg_term  

		self.J_Theta = J

		return J 
		
	def build_J_xent_w_scaled_reg(self,y=None,lambda_val=1.0):
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
		# m=number of samples, i.e. total "batch" size
		X=self.X
		m = tf.cast( tf.shape( X )[0], tf.float32)
		
		loss = self.build_J_xent(y)
		
		Thetas_only = self.DNN_model.__get_state__()['Thetas']
		
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
			self.X_data = X_data
		else:
			X_data=self.X_data

		if y_data is not None:
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
