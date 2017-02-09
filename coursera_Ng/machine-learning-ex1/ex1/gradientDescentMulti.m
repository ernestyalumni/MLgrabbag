function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

features = size(X,2);  # number of features

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    temp_vals = zeros(1, features) ;
    
    predicted_vals = X * theta;

    res = predicted_vals - y; # res for residual

    for feature = 1:features 
      dres = res .* X(:,feature) ;
      temp = theta(feature,1) - alpha * mean( dres ) ;
      temp_vals(1,feature) = temp ;
    end  
    
    theta = temp_vals' ; # remember to take the transpose
      
    





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
