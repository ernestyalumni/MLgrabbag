function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % size(X) should yield something like (97,2), 97 training data pts., 2 features 
    
    predicted_vals = X * theta;

    res = predicted_vals - y; # res for residual

    % split up the X into each x_j, each j
    X_1 = X(:,1);
    X_2 = X(:,2); 
    
    % it's the partial derivative with respect to each x_j; element wise multiplication with .*
    dres_1 = res .* X_1; 
    dres_2 = res .* X_2;
    
    temp_1 = theta(1,1) - alpha * mean( dres_1 );
    temp_2 = theta(2,1) - alpha * mean( dres_2) ;
    
    temp = [ temp_1, temp_2 ]; % remember to take the transpose to get a column vector, not a row vector
    theta = temp';  % remember to take the transpose with '
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
