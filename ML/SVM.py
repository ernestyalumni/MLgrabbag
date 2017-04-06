"""
@file SVM.py
@brief SVM, Support Vector Machines

@author Ernest Yeung
@email ernestyalumni dot gmail dot com

"""  
import theano
import numpy as np
import theano.tensor as T
from theano import sandbox

# other people had this problem too: it's a problem INHERENT with Python
# max recusion limit #689 
# https://github.com/Theano/Theano/issues/689
import sys
sys.setrecursionlimit(10000)

def rbf(Xj,Xi,sigma):
    rbf = T.exp(-((Xj-Xi)**2).sum()/(np.float32(2.*sigma**2)))
    return rbf

"""    
def rbf_step(Xj,yj,lambda_multj,  # sequences to iterate over
			cumulative_sum,   # previous iteration 
			Xi,yi,lambda_multi,sigma):  # non-sequences needed at each iteration; they don't change
    W_i = lambda_multi*yi*lambda_multj*yj*rbf(Xj,Xi,sigma)
    return cumulative_sum + W_i
"""

class SVM(object):
	""" SVM - Support Vector Machines 
	"""
	def __init__(self,X,y,m,C,sigma,alpha):
		assert m == X.shape[0] and m == y.shape[0]
#		self._C_val = np.float32(C)
		self.C = theano.shared( np.float32(C)) # need to be made into a theano shared variable for upper bound check
				
		self._sigma_val = np.float32(sigma)
		self._alpha_val = np.float32(alpha)
		
		self._m_val = np.int32(m)

#        self._m = theano.shared( np.int32(m))
		
		self.X = theano.shared( X.astype(theano.config.floatX) )
		self.y = theano.shared( y.astype(theano.config.floatX) )
		self.lambda_mult = theano.shared( np.random.rand(m).astype(theano.config.floatX) ) # lambda Lagrange multipliers
		
#		self.sigma = theano.shared( self._alpha_val)
		
	def build_W(self):
		m = self._m_val
		
		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult
		
		sigma=self._sigma_val

		def rbf_step(Xj,yj,lambda_multj,  # sequences to iterate over
			cumulative_sum,   # previous iteration 
			Xi,yi,lambda_multi):  # non-sequences needed at each iteration; they don't change
			W_i = lambda_multi*yi*lambda_multj*yj*rbf(Xj,Xi,sigma)
			return cumulative_sum + W_i
	
		
		W_00 = theano.shared(np.float32(0.))
		output,update=theano.reduce(fn=rbf_step,
									sequences=[X,y,lambda_mult],outputs_info=[W_00],
									non_sequences=[X[0],y[0],lambda_mult[0]])
		updates=[update,]
		
		for i in range(1,m):
			W_i0= theano.shared(np.float32(0.))
			outputi,updatei=theano.reduce(fn=rbf_step,
										sequences=[X,y,lambda_mult],outputs_info=[W_i0],
										non_sequences=[X[i],y[i],lambda_mult[i]])
			output +=outputi
			updates.append(updatei)
		
		output *= np.float32(0.5)
		
		output -= lambda_mult.sum()
		
		self.W = sandbox.cuda.basic_ops.gpu_from_host( output )
		
		return output, updates  
		
	def build_update(self, alpha=0.01, beta=0.0):
		W = self.W
		lambda_mult=self.lambda_mult
		y=self.y
		C = self.C
		lower_bound = theano.shared(np.float32(0.0))
		
		updates = build_gradDescent_step(W, lambda_mult, alpha,beta)
		updatelambda_mult = updates[1]  # \Longleftrightarrow  <<===>> \lambda_i'(t+1)
		
		updatelambda_mult = updatelambda_mult - T.dot(y,updatelambda_mult)/T.dot(y,y) * y 	# Longleftrightarrow <<===>> \lambda_i''(t+1)
		
		# use theano.tensor.switch because we need an elementwise comparison 
		# if \lambda_I''(t+1)> C, C
		updatelambda_mult = T.switch( T.lt( C , updatelambda_mult), C, updatelambda_mult)
		updatelambda_mult = T.switch( T.lt( updatelambda_mult,lower_bound), lower_bound, updatelambda_mult)
		
		updatelambda_mult = sandbox.cuda.basic_ops.gpu_from_host( updatelambda_mult)

		updatefunction = theano.function(inputs=[], 
										outputs = W,
										updates=[(lambda_mult, updatelambda_mult)])

		self._update_lambda_mult_graph = updatelambda_mult
		self.update_function = updatefunction

		return updatelambda_mult, updatefunction

	def train_model_full(self, max_iters=250):
		train_errors = np.ndarray(max_iters)
		
		update_function = self.update_function
		
		for iter in range( max_iters):
			error = 0.

			W_train = update_function()
#			train_errors[iter] = W_train

			if np.isnan( W_train ) or np.isinf( W_train):
#				print('bad cost detected: ', J_train )
				error += np.array( 1.)
			else: 
				error += np.array( W_train)

			train_errors[iter] = error


	
		return train_errors
	

	def build_b(self):
		m = np.float32( self._m_val )
		
		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val
		
		def rbf_step(Xj,yj,lambda_multj,  # sequences to iterate over
			cumulative_sum,   # previous iteration 
			Xi):  # non-sequences needed at each iteration; they don't change
			b_i = lambda_multj*yj*rbf(Xj,Xi,sigma)
			return cumulative_sum + b_i

		b_00 = theano.shared(np.float32(0.))
		output,update=theano.reduce(fn=rbf_step,
									sequences=[X,y,lambda_mult],outputs_info=[b_00],
									non_sequences=[X[0]])
		updates=[update,]
		
		for i in range(1,m):
			b_i0= theano.shared(np.float32(0.))
			outputi,updatei=theano.reduce(fn=rbf_step,
										sequences=[X,y,lambda_mult],outputs_info=[b_i0],
										non_sequences=[X[i]])
			output +=outputi
			updates.append(updatei)

#		output = np.float32(1.)/m * ( y.sum() - np.float32(-1.) * output )
		output = np.float32(1./m) * ( y.sum() - output )
		
		self.b = output
		
		return output, updates

	def make_predict(self,X_test_val):
		m = self._m_val
		
		X_test_val = X_test_val.astype(theano.config.floatX)
		X_test = theano.shared( X_test_val )

		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val
		
		b = self.b
	
		def rbf_step(Xj,yj,lambda_multj,  # sequences to iterate over
			cumulative_sum,   # previous iteration 
			Xi ):  # non-sequences needed at each iteration; they don't change
			y_pred = lambda_multj*yj*rbf(Xj,Xi,sigma)
			return cumulative_sum + y_pred
	
		y_pred_var = theano.shared(np.float32(0.))
	
		output,update=theano.reduce(fn=rbf_step,
									sequences=[X,y,lambda_mult],outputs_info=[y_pred_var],
									non_sequences=[X_test])

		output += b

		y_pred_function = theano.function(inputs=[], outputs = output)
		y_pred_val = y_pred_function()
		return y_pred_val, y_pred_function

	def make_predictions(self,X_pred_vals):
#		m = self._m_val

		m_pred = X_pred_vals.shape[0]  # number of examples X to make predictions on 
		
		X_pred_vals = X_pred_vals.astype(theano.config.floatX)
		X_pred = theano.shared( X_pred_vals )
		X_pred_var = theano.shared( X_pred_vals[0] )


		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val
		
		b = self.b
	
		def rbf_step(Xj,yj,lambda_multj,  # sequences to iterate over
			cumulative_sum,   # previous iteration 
			Xi ):  # non-sequences needed at each iteration; they don't change
			y_pred = lambda_multj*yj*rbf(Xj,Xi,sigma)  
			return cumulative_sum + y_pred
	
		predictions_lst = []	

		# define a general prediction formula, 
		y_pred_var = theano.shared(np.float32(0.))  # accumulate predictions of y in this shared variable
		output,update=theano.reduce(fn=rbf_step,
									sequences=[X,y,lambda_mult],outputs_info=[y_pred_var],
									non_sequences=[X_pred_var])
		output += b
		output = sandbox.cuda.basic_ops.gpu_from_host( output )	# output here corresponds to yhat(X), X\equiv Xi, and is ready for substituting in values through givens parameter		


		for i in range(m_pred):
			y_pred_i = theano.shared(np.float32(0.))  # accumulate predictions of y in this shared variable
			X_pred_i = X_pred[i]  # ith example of input data X to make predictions on 
			
			y_pred_function = theano.function(inputs=[],outputs=output,givens=[(X_pred_var,X_pred_i),(y_pred_var,y_pred_i)])
			y_pred_val = y_pred_function()
			predictions_lst.append( y_pred_val )


	
		return predictions_lst


class SVM_serial(object):
	""" SVM_serial - Support Vector Machines, with W computed with for loops 
	"""
	def __init__(self,X,y,m,C,sigma,alpha):
		assert m == X.shape[0] and m == y.shape[0]
		self.C = theano.shared( np.float32(C)) # need to be made into a theano shared variable for upper bound check
				
		self._sigma_val = np.float32(sigma)
		self._alpha_val = np.float32(alpha)
		
		self._m_val = np.int32(m)
		
		self.X = theano.shared( X.astype(theano.config.floatX) )
		self.y = theano.shared( y.astype(theano.config.floatX) )
		self.lambda_mult = theano.shared( np.random.rand(m).astype(theano.config.floatX) ) # lambda Lagrange multipliers
		
		
	def build_W(self):
		m = self._m_val
		
		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult
		
		sigma=self._sigma_val

		# begin life at 0; W is an accumulator of scalar values, in \mathbb{R}
		W = theano.shared(np.float32(0.))
		
		for i in range(m):
			W_i = theano.shared(np.float32(0.))
			Xi = X[i]
			yi = y[i]
			lambda_multi = lambda_mult[i]
			for j in range(m):
				Xj=X[j]
				yj=y[j]
				lambda_multj=lambda_mult[j]
				W_i += lambda_multi*yi*lambda_multj*yj*rbf(Xj,Xi,sigma)	
			W += W_i

		
		W *= np.float32(0.5)
		
		W -= lambda_mult.sum()
		
		self.W = sandbox.cuda.basic_ops.gpu_from_host( W )
		
		return W
		
	def build_update(self, alpha=0.01, beta=0.0):
		W = self.W
		lambda_mult=self.lambda_mult
		y=self.y
		C = self.C
		lower_bound = theano.shared(np.float32(0.0))
		
		updates = build_gradDescent_step(W, lambda_mult, alpha,beta)
		updatelambda_mult = updates[1]  # \Longleftrightarrow  <<===>> \lambda_i'(t+1)
		
		updatelambda_mult = updatelambda_mult - T.dot(y,updatelambda_mult)/T.dot(y,y) * y 	# Longleftrightarrow <<===>> \lambda_i''(t+1)
		
		# use theano.tensor.switch because we need an elementwise comparison 
		# if \lambda_I''(t+1)> C, C
		updatelambda_mult = T.switch( T.lt( C , updatelambda_mult), C, updatelambda_mult)
		updatelambda_mult = T.switch( T.lt( updatelambda_mult,lower_bound), lower_bound, updatelambda_mult)
		
		updatelambda_mult = sandbox.cuda.basic_ops.gpu_from_host( updatelambda_mult)

		updatefunction = theano.function(inputs=[], 
										outputs = W,
										updates=[(lambda_mult, updatelambda_mult)])

		self._update_lambda_mult_graph = updatelambda_mult
		self.update_function = updatefunction

		return updatelambda_mult, updatefunction

	def train_model_full(self, max_iters=250):
		train_errors = np.ndarray(max_iters)
		
		update_function = self.update_function
		
		for iter in range( max_iters):
			error = 0.

			W_train = update_function()

			if np.isnan( W_train ) or np.isinf( W_train):
				error += np.array( 1.)
			else: 
				error += np.array( W_train)

			train_errors[iter] = error


	
		return train_errors
	

	def build_b(self):
		m = self._m_val 
		
		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val

		b = theano.shared(np.float32(0.))
		
		for i in range(m):
			b_i = theano.shared(np.float32(0.))
			
			Xi = X[i]
			
			for j in range(m):
				Xj=X[j]
				yj=y[j]
				lambda_multj=lambda_mult[j]
				b_i += lambda_multj*yj*rbf(Xj,Xi,sigma)
			b += b_i
		
		b = np.float32(1./m)*(y.sum()-b)
		b = sandbox.cuda.basic_ops.gpu_from_host( b)		

		self.b = b
		
		return b

	def make_predict(self,X_test_val):
		m = self._m_val
		
		X_test_val = X_test_val.astype(theano.config.floatX)
		X_test = theano.shared( X_test_val )

		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val
		
		b = self.b
	
		yhat = theano.shared(np.float32(0.))
		for j in range(m):
			Xj=X[j]
			yj=y[j]
			lambda_multj=lambda_mult[j]
			yhat += lambda_multj*yh*rbf(Xj,X_test,sigma)
		yhat_function=theano.function(inputs=[],outputs=yhat)
		yhat_val=yhat_function()

		return yhat_val, yhat_function

	def make_predictions(self,X_pred_vals):
		m = self._m_val

		m_pred = X_pred_vals.shape[0]  # number of examples X to make predictions on 
		
		X_pred_vals = X_pred_vals.astype(theano.config.floatX)
		X_pred = theano.shared( X_pred_vals )

		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val
		
		b = self.b

		predictions_lst = []
		for example_idx in range(m_pred):
			Xexample = X_pred[example_idx]

			yhat = theano.shared(np.float32(0.))
			for j in range(m):
				Xj=X[j]
				yj=y[j]
				lambda_multj=lambda_mult[j]
				yhat += lambda_multj*yj*rbf(Xj,Xexample,sigma)
			yhat_function = theano.function(inputs=[],outputs=yhat)
			yhat_val = yhat_function()
			predictions_lst.append(yhat_val)

	
		return predictions_lst
		
class SVM_parallel(object):
	""" SVM - Support Vector Machines in parallel (no for loops, only theano scan)
	"""
	def __init__(self,X,y,m,C,sigma,alpha):
		assert m == X.shape[0] and m == y.shape[0]
		self.C = theano.shared( np.float32(C)) # need to be made into a theano shared variable for upper bound check
				
		self._sigma_val = np.float32(sigma)
		self._alpha_val = np.float32(alpha)
		
		self._m_val = np.int32(m)
		
		self.X = theano.shared( X.astype(theano.config.floatX) )
		self.y = theano.shared( y.astype(theano.config.floatX) )
		self.lambda_mult = theano.shared( np.random.rand(m).astype(theano.config.floatX) ) # lambda Lagrange multipliers
		
		
	def build_W(self):
		m = self._m_val
		
		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult
		
		sigma=self._sigma_val

		W = theano.shared(np.float32(0.))

		
		def i_step(Xi,yi,lambda_multi, # sequences to iterate over
					cumulative_sum,  	# previous iteration
					X,y,lambda_mult):  # non-sequences needed at each iteration; they don't change
			
			# j step, calculating W_i
			def j_step(Xj,yj,lambda_multj, # sequences to iterate over
						cumulative_sum, 	# previous iteration
						Xi): # non-sequence needed at each jth iteration, it doesn't change
				W_ij = lambda_multj*yj*rbf(Xj,Xi,sigma)
				return cumulative_sum + W_ij
			
			W_i0=theano.shared(np.float32(0.))
			output,update=theano.reduce(fn=j_step,
										sequences=[X,y,lambda_mult],outputs_info=[W_i0],
										non_sequences=[Xi] )
			# output at this point, symbolically represents W_i
			W_i = lambda_multi * yi * output
			return cumulative_sum + W_i
			
		output,update=theano.reduce(fn=i_step,
									sequences=[X,y,lambda_mult],outputs_info=[W],
									non_sequences=[X,y,lambda_mult])
									
		output *= np.float32(0.5)
		
		output -= lambda_mult.sum()
		
		self.W = sandbox.cuda.basic_ops.gpu_from_host( output )
		
		return output, update  
		
	def build_update(self, alpha=0.01, beta=0.0):
		W = self.W
		lambda_mult=self.lambda_mult
		y=self.y
		C = self.C
		lower_bound = theano.shared(np.float32(0.0))
		
		updates = build_gradDescent_step(W, lambda_mult, alpha,beta)
		updatelambda_mult = updates[1]  # \Longleftrightarrow  <<===>> \lambda_i'(t+1)
		
		updatelambda_mult = updatelambda_mult - T.dot(y,updatelambda_mult)/T.dot(y,y) * y 	# Longleftrightarrow <<===>> \lambda_i''(t+1)
		
		# use theano.tensor.switch because we need an elementwise comparison 
		# if \lambda_I''(t+1)> C, C
		updatelambda_mult = T.switch( T.lt( C , updatelambda_mult), C, updatelambda_mult)
		updatelambda_mult = T.switch( T.lt( updatelambda_mult,lower_bound), lower_bound, updatelambda_mult)
		
		updatelambda_mult = sandbox.cuda.basic_ops.gpu_from_host( updatelambda_mult)

		updatefunction = theano.function(inputs=[], 
										outputs = W,
										updates=[(lambda_mult, updatelambda_mult)])

		self._update_lambda_mult_graph = updatelambda_mult
		self.update_function = updatefunction

		return updatelambda_mult, updatefunction

	def train_model_full(self, max_iters=250):
		train_errors = np.ndarray(max_iters)
		
		update_function = self.update_function
		
		for iter in range( max_iters):
			error = 0.

			W_train = update_function()

			if np.isnan( W_train ) or np.isinf( W_train):
				error += np.array( 1.)
			else: 
				error += np.array( W_train)

			train_errors[iter] = error
	
		return train_errors
	

	def build_b(self):
		m = np.float32( self._m_val )
		
		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val

		def i_step(Xi, # sequences to iterate over
					cumulative_sum,  	# previous iteration
					X,y,lambda_mult):  # non-sequences needed at each iteration; they don't change
			
			# j step, calculating W_i
			def j_step(Xj,yj,lambda_multj, # sequences to iterate over
						cumulative_sum, 	# previous iteration
						Xi): # non-sequence needed at each jth iteration, it doesn't change
				b_ij = lambda_multj*yj*rbf(Xj,Xi,sigma)
				return cumulative_sum + b_ij
			
			b_i0=theano.shared(np.float32(0.))
			output,update=theano.reduce(fn=j_step,
										sequences=[X,y,lambda_mult],outputs_info=[b_i0],
										non_sequences=[Xi] )
			# output at this point, symbolically represents b_i
			
			return cumulative_sum + output

		b = theano.shared(np.float32(0.))
		output,update=theano.reduce(fn=i_step,
									sequences=[X],outputs_info=[b],
									non_sequences=[X,y,lambda_mult])

		output = np.float32(1./m) * ( y.sum() - output )
		
		self.b = sandbox.cuda.basic_ops.gpu_from_host( output )
		
		return output, update

	def make_predict(self,X_test_val):
		m = self._m_val
		
		X_test_val = X_test_val.astype(theano.config.floatX)
		X_test = theano.shared( X_test_val )

		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val
		
		b = self.b
	
		def rbf_step(Xj,yj,lambda_multj,  # sequences to iterate over
			cumulative_sum,   # previous iteration 
			Xi ):  # non-sequences needed at each iteration; they don't change
			y_pred = lambda_multj*yj*rbf(Xj,Xi,sigma)
			return cumulative_sum + y_pred
	
		y_pred_var = theano.shared(np.float32(0.))
	
		output,update=theano.reduce(fn=rbf_step,
									sequences=[X,y,lambda_mult],outputs_info=[y_pred_var],
									non_sequences=[X_test])

		output += b

		y_pred_function = theano.function(inputs=[], outputs = output)
		y_pred_val = y_pred_function()
		return y_pred_val, y_pred_function

	def make_predictions(self,X_pred_vals):

		m_pred = X_pred_vals.shape[0]  # number of examples X to make predictions on 
		
		X_pred_vals = X_pred_vals.astype(theano.config.floatX)
		X_pred = theano.shared( X_pred_vals )
		X_pred_var = theano.shared( X_pred_vals[0] )


		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val
		
		b = self.b
	
		def rbf_step(Xj,yj,lambda_multj,  # sequences to iterate over
			cumulative_sum,   # previous iteration 
			Xi ):  # non-sequences needed at each iteration; they don't change
			y_pred = lambda_multj*yj*rbf(Xj,Xi,sigma)  
			return cumulative_sum + y_pred
	
		predictions_lst = []	

		# define a general prediction formula, 
		y_pred_var = theano.shared(np.float32(0.))  # accumulate predictions of y in this shared variable
		output,update=theano.reduce(fn=rbf_step,
									sequences=[X,y,lambda_mult],outputs_info=[y_pred_var],
									non_sequences=[X_pred_var])
		output += b
		output = sandbox.cuda.basic_ops.gpu_from_host( output )	# output here corresponds to yhat(X), X\equiv Xi, and is ready for substituting in values through givens parameter		


		for i in range(m_pred):
			y_pred_i = theano.shared(np.float32(0.))  # accumulate predictions of y in this shared variable
			X_pred_i = X_pred[i]  # ith example of input data X to make predictions on 
			
			y_pred_function = theano.function(inputs=[],outputs=output,givens=[(X_pred_var,X_pred_i),(y_pred_var,y_pred_i)])
			y_pred_val = y_pred_function()
			predictions_lst.append( y_pred_val )


	
		return predictions_lst

	def make_predictions_parallel(self,X_pred_vals):
		""" make_predictions_parallel - make predictions on given X_pred_vals, X values to predict on, with no for loops, only theano scan (hence named parallel)
		"""
		X_pred_vals = X_pred_vals.astype(theano.config.floatX)
		X_pred = theano.shared( X_pred_vals )

		X=self.X
		y=self.y
		lambda_mult=self.lambda_mult

		sigma=self._sigma_val
		
		b = self.b

		
		def step(Xi, # sequence to iterate over
				X,y,lambda_mult,b): # non-recurrent non-sequences needed at each iteration; they don't change
			def j_step(Xj,yj,lambda_multj, # sequences to iterate over the sum from j=1...m
						cumulative_sum, # previous iteration
						Xi): # non-sequence needed at each iteration for the summation
				yhat_j = lambda_multj*yj*rbf(Xj,Xi,sigma)
				return cumulative_sum + yhat_j
			
			yhat_0 = theano.shared(np.float32(0.))
			output,update=theano.reduce(fn=j_step,
										sequences=[X,y,lambda_mult],outputs_info=[yhat_0],
										non_sequences=[Xi] )
			#output now represents the summation in the formula for yhat, but without the intercept b
			output += b
			return output
		
		# no need for outputs_info parameter since previous iterations not needed
		output,update=theano.scan(fn=step,
								sequences=[X_pred], non_sequences=[X,y,lambda_mult,b])
		output = sandbox.cuda.basic_ops.gpu_from_host(output)
		
		
		predictions_function = theano.function(inputs=[],outputs=output)
		predictions_vals = predictions_function()
		self._yhat = theano.shared( predictions_vals ) # added this line later


		return predictions_vals, predictions_function  # originally
#		return predictions_vals, predictions_function,output
		
		
	# cf. http://www.datascienceassn.org/sites/default/files/Predicting%20good%20probabilities%20with%20supervised%20learning.pdf
	# Platt Calibration	
	def make_prob_Pratt(self,y,alpha=0.01,training_steps=10000):
		""" make_prob - make probabilities, according to Pratt scaling 
		@type y : numpy array of length m, of binary values 0,1 
		@param y : actual class the example is in

		@type alpha : float
		@param alpha : "learning rate" for gradient descent

		WARNING: make sure this inputted y is in 0,1 representation, NOT -1,1 representation of which (binary) class it's in 

		returns the actual values as a (numerical) array and theano expression (graph)
		"""
		alpha = np.float32(alpha)
		y_sh = theano.shared( y.astype(theano.config.floatX ) )
		
#		predicted_vals,predicting_function,output_graph = self.make_predictions_parallel( X_pred_vals)
		yhat = self._yhat 
		A = theano.shared( np.float32( np.random.rand() ) )
		B = theano.shared( np.float32( np.random.rand() ) )
		Prob_1_given_yhat = np.float32(1.)/(np.float32(1.)+ T.exp(A*yhat +B)) # P(y=1|f) = 1/(1+exp(Af+B)), should be length (size or dimension) of m 
		
		costfunctional = T.nnet.binary_crossentropy( Prob_1_given_yhat, y_sh).mean()
		
		DA, DB = T.grad(costfunctional, [A,B])  # the gradient of costfunctional, with respect to A,B, respectively

		updateA = sandbox.cuda.basic_ops.gpu_from_host(A-alpha*DA)
		updateB = sandbox.cuda.basic_ops.gpu_from_host(B-alpha*DB)

		train = theano.function(inputs=[],outputs=[Prob_1_given_yhat, costfunctional],
#								updates=[(A,updateA),(B,B-alpha*DB)],name="train")  # on the CPU
								updates=[(A,updateA),(B,updateB)],name="train")  # on the GPU

		probabilities = theano.function(inputs=[], outputs=Prob_1_given_yhat,name="probabilities")
		
		for i in range(training_steps):
			pred,err = train()
		
		probabilities_vals = probabilities()
		return probabilities_vals, Prob_1_given_yhat
		
		
def build_gradDescent_step( W, lambda_mult, alpha =0.01, beta= 0.0):
	""" build_gradDescent_step - gradient Descent (with momentum), but from build_cost_functional for the J

	INPUT/PARAMETERS:
	=================
	@param W     : cost function (from build_cost_function)
	
	@type lambda_mult  : theano shared variable 
	@param lambda_mult : Lagrange multipliers (to optimize over) 
	
	@param alpha : learning rate
	
	@param beta  : "momentum" constant (parameter)

	RETURN(S)
	=========
	@type updateThetas, gradientDescent_step : tuple of (list of theano symbolic expression, theano function)

	"""
	alpha = np.float32(alpha)
	beta = np.float32(beta) 
	
	updatelambda_mult = lambda_mult - alpha * T.grad( W,lambda_mult) + beta * lambda_mult

	updates= (lambda_mult, updatelambda_mult)
	
	return updates
	
	
########################################################################
########### END of necessary class and function definitions ############
########################################################################	
	
if __name__ == "__main__":
	print("In main")	
	m=4
	d=2
	X_val=np.arange(2,m*d+2).reshape(m,d).astype(theano.config.floatX) 
	y_val=np.random.randint(2,size=m).astype(theano.config.floatX)
	lambda_mult_val = np.random.rand(m).astype(theano.config.floatX)
	test_SVM=SVM(X_val,y_val,m,1.0,1.0,0.01)
	test_SVM.build_W()
	# This following step may take a while
	test_SVM.build_update()
	test_SVM.train_model_full()
	test_SVM.build_b()
