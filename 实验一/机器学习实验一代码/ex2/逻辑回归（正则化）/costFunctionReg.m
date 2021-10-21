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

Sum = 0;
for i=1:m
    sum = 0;
    %得到h_theta的值
    sum = X(i,:) * theta;
    sum = sigmoid(sum);
    Sum = Sum + ((-1)*y(i)*log(sum)-(1-y(i))*log(1-sum));
end
J = Sum/m;
Q = 0;
for i=1:size(theta,1)
  Q = Q + theta(i,1) * theta(i,1);  
end
J = J + (lambda/(2*m))*Q;

for i=1:size(theta,1)
    Sum = 0;
    for j=1:m
        sum = 0;
        %get the value of h_theta
        for k=1:size(theta,1)
            sum = sum + X(j,k) * theta(k,1);
        end
        sum = sigmoid(sum);
        Sum = Sum + (sum-y(j))*X(j,i);       
    end
    % special case
    if i == 1
        grad(i,1) = Sum/m;  
    else
        grad(i,1) = Sum/m+(lambda/m)*theta(i,1); 
    end 
end




% =============================================================

end
