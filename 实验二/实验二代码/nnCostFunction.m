function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%对每一个X的样例，添加偏置项
X = [ones(size(X,1),1) X];  %现在X是一个5000*401的矩阵
z2 = X * Theta1';  %z2是一个5000*25的矩阵

a2 = sigmoid(z2);  %a2 = g(z2)是一个5000*25的矩阵

a2 = [ones(size(a2,1),1) a2];  %添加偏置项，是一个5000*26的矩阵

z3 = a2 * Theta2';  %是一个5000*10的矩阵

h = sigmoid(z3);  %得到h_theta矩阵，5000*10的矩阵

y_recoded = zeros(size(X,1),num_labels);  %y重新编码矩阵，是一个5000*10的矩阵

for i = 1:size(X,1)
   y_recoded(i,y(i,1)) = 1; 
end

sum = 0;
for i = 1:size(X,1)
    res1 = y_recoded(i,:) * log(h(i,:)');
    res1 = -res1;
    res2 = (1 - y_recoded(i,:)) * log(1 - h(i,:)');
    res2 = -res2;
    
    sum = sum + res1 + res2;
end

J = sum/size(X,1);

sum = 0;
for i = 1:size(Theta1,1)
    for j = 2:size(Theta1,2)
        sum = sum + Theta1(i,j) * Theta1(i,j);
    end
end
for i = 1:size(Theta2,1)
    for j = 2:size(Theta2,2)
        sum = sum + Theta2(i,j) * Theta2(i,j);
    end
end
sum = sum * (lambda/(2*m));

J = J + sum;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%%%% Back Propagation
% h: 5000*10
% y_recoded: 5000*10
% z2: 5000*25
delta3 = h - y_recoded;	    % 5000*10
delta2 = Theta2'* delta3';  % Theta2':26*10 delta3':10*5000 乘积结果为26*5000
delta2 = delta2(2:end,:);	% 截掉第一行成为25*5000的矩阵
delta2 = delta2.*sigmoidGradient(z2)';	%(25*5000).*(25*5000)得到25*5000的误差矩阵

Delta1 = zeros(size(Theta1));  % 25*401 
Delta2 = zeros(size(Theta2));  % 10*26

Delta1 = Delta1 + delta2 * X; % (25*5000) * (5000*401) 得到25*401的矩阵
Delta2 = Delta2 + delta3' * a2; % (10*5000) * (5000*26) 得到10*26的矩阵

Theta1_grad = Delta1 / m; % 25*401的矩阵
Theta2_grad = Delta2 / m; % 10*26的矩阵

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Reg1 = lambda/m.*Theta1(:,2:end);  % 25*400
Reg2 = lambda/m.*Theta2(:,2:end);  % 10*25
Reg1 = [zeros(size(Theta1,1),1),Reg1]; %j>=1
Reg2 = [zeros(size(Theta2,1),1),Reg2]; %j>=1
Theta1_grad = Theta1_grad + Reg1;
Theta2_grad = Theta2_grad + Reg2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
