function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
% m is the size of sample
Sum = 0;
for i=1:m
    sum = 0;
    %get the value of h_theta
    for j=1:size(theta,1)
        sum = sum + X(i,j) * theta(j,1);
    end
    sum = sigmoid(sum);
    Sum = Sum + ((-1)*y(i)*log(sum)-(1-y(i))*log(1-sum));
end
J = Sum/m;
% =============================================================
for i=1:size(theta,1)
    Sum = 0;
    for j=1:m
        sum = 0;
        %calculate the value of h_theta
        for k=1:size(theta,1)
            sum = sum + X(j,k) * theta(k,1);
        end
        sum = sigmoid(sum);
        Sum = Sum + (sum-y(j))*X(j,i);       
    end
    grad(i,1) = Sum/m;  
end


end
