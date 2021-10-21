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
%��ÿһ��X�����������ƫ����
X = [ones(size(X,1),1) X];  %����X��һ��5000*401�ľ���
z2 = X * Theta1';  %z2��һ��5000*25�ľ���

a2 = sigmoid(z2);  %a2 = g(z2)��һ��5000*25�ľ���

a2 = [ones(size(a2,1),1) a2];  %���ƫ�����һ��5000*26�ľ���

z3 = a2 * Theta2';  %��һ��5000*10�ľ���

h = sigmoid(z3);  %�õ�h_theta����5000*10�ľ���

y_recoded = zeros(size(X,1),num_labels);  %y���±��������һ��5000*10�ľ���

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
delta2 = Theta2'* delta3';  % Theta2':26*10 delta3':10*5000 �˻����Ϊ26*5000
delta2 = delta2(2:end,:);	% �ص���һ�г�Ϊ25*5000�ľ���
delta2 = delta2.*sigmoidGradient(z2)';	%(25*5000).*(25*5000)�õ�25*5000��������

Delta1 = zeros(size(Theta1));  % 25*401 
Delta2 = zeros(size(Theta2));  % 10*26

Delta1 = Delta1 + delta2 * X; % (25*5000) * (5000*401) �õ�25*401�ľ���
Delta2 = Delta2 + delta3' * a2; % (10*5000) * (5000*26) �õ�10*26�ľ���

Theta1_grad = Delta1 / m; % 25*401�ľ���
Theta2_grad = Delta2 / m; % 10*26�ľ���

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
