function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
if (m==0) 
  m=1;
end

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predicted_val = X * theta ; 
res = predicted_val - y; 
J = ( res' * res )/m * 0.5 ;

% regularized term
theta1 = theta(2:end,:) ; % should not regularize the \theta_0 term, 
reg_term = theta1' * theta1 ;
reg_term = lambda / (2 * m) * reg_term ;

J += reg_term ; 

res = predicted_val - y ; 

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








% =========================================================================

grad = grad(:);

end
