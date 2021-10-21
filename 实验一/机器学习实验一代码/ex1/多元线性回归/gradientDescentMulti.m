function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta1 = theta;
for iter = 1:num_iters
    % n-1 is the number of variables
    % m is the size of samples
    n = size(X,2); 
    for i=1:n
        Sum = 0;
        for j=1:m
            sum = X(j,:) * theta;
            Sum = Sum + (sum - y(j)) * X(j,i);
        end
        theta1(i,1) = theta(i,1) - alpha * (1/m) * Sum;
    end
    theta = theta1;
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
