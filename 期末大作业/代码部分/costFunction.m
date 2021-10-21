function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

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
