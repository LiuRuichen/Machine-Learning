function J = computeCost(X, y, theta)  %theta³õÊ¼ÖµÎª[0,0]
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
tmp1 = theta(1,1);
tmp2 = theta(2,1);

sum = 0;
for j=1:m
    sum = sum+(tmp1+tmp2*X(j,2)-y(j))*(tmp1+tmp2*X(j,2)-y(j));
end

J = sum/(2*m);
    


% =========================================================================

end
