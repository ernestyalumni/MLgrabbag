function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%predicted_vals = X * theta ;
%predicted_vals = sigmoid( predicted_vals ) ;

%interpolation = -y .* log( predicted_vals ) - (1. - y) .* log( 1. - predicted_vals ) ;

%interpolation = -y' *  

%J = (1/m) .* sum( interpolation ) ;



% regularized term
%theta1 = theta(2:end,:) ; 
%reg_term = theta1' * theta1 ;
%reg_term = lambda / (2 * m) * reg_term ;

%J += reg_term ; 

%res = predicted_vals - y ; 

%dJ = res .* X ;

%dJ = X' * res ;

%grad = mean( dJ ) ; 

% Note that for GNU Octave, 
% "If x is a matrix, compute the mean for each column and return them in a row vector: 
% cf. https://www.gnu.org/software/octave/doc/interpreter/Descriptive-Statistics.html


% regularization term for gradient : grad_reg_term
%grad_reg_term = zeros(size(theta)) ;
%grad_reg_term(2:end,:) = theta(2:end,:) ;
%grad_reg_term *= lambda / m ;

%grad = dJ +  grad_reg_term' ;



n=length(theta);
z=X*theta;
h=sigmoid(z);
logisf=(-y)'*log(h)-(1-y)'*log(1-h);
thetas=theta(2:n,1);
J=((1/m).*sum(logisf))+(lambda/(2*m)).*sum(thetas.^2);

% Regularized
temp=theta;
temp(1)=0;
grad=(1/m).*(X'*(h-y)+lambda.*temp);






% =============================================================

grad = grad(:);

end
