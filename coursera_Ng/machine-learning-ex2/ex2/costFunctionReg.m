function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predicted_vals = X * theta ;
predicted_vals = sigmoid( predicted_vals ) ;

interpolation = -y .* log( predicted_vals ) - ( 1 - y ) .* log( 1 - predicted_vals ) ;

J = mean( interpolation ) ;

% regularized term
theta1 = theta(2:end,:) ; 
reg_term = theta1' * theta1 ;
reg_term = lambda / (2 * m) * reg_term ;

J += reg_term ; 

res = predicted_vals - y ; 

dJ = res .* X ;

grad = mean( dJ ) ; 
% Note that for GNU Octave, 
% "If x is a matrix, compute the mean for each column and return them in a row vector: 
% cf. https://www.gnu.org/software/octave/doc/interpreter/Descriptive-Statistics.html


% regularization term for gradient : grad_reg_term
grad_reg_term = zeros(size(theta)) ;
grad_reg_term(2:end,:) = theta(2:end,:) ;
grad_reg_term *= lambda / m ;

grad += grad_reg_term' ;




% =============================================================

end
