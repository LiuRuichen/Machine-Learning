%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData2(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic 
%  regression to classify the data points. 
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 0;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);
ini2 = initial_theta;
ini3 = initial_theta;

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;
lambda2 = 0;
lambda3 = 100;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 225);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

[theta2, J2, exit_flag2] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda2)), ini2, options);

[theta3, J3, exit_flag3] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda3)), ini3, options);
% disp(theta);
% disp(theta2);

figure;
% Plot Boundary
subplot(1,3,1);
plotDecisionBoundary(theta, X, y);
title(sprintf('lambda = %g', lambda));
legend('y = 1', 'y = 0', 'Decision boundary');
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');
hold off;

subplot(1,3,2);
plotDecisionBoundary(theta2, X, y);
title(sprintf('lambda = %g', lambda2));
legend('y = 1', 'y = 0', 'Decision boundary');
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');
hold off;

subplot(1,3,3);
plotDecisionBoundary(theta3, X, y);
title(sprintf('lambda = %g', lambda3));
legend('y = 1', 'y = 0', 'Decision boundary');
xlabel('Microchip Test 1');
ylabel('Microchip Test 2');
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('Here is a sample for predicting:\n');
x = [0.3,0.6];
tmp = 1;
degree = 6;
out = [];
for i = 1:degree
    for j = 0:i
        out(tmp) = (x(1)^(i-j))*(x(2)^j);
        tmp = tmp + 1;
    end
end
out = [1 out];
a = '%';
disp(['When test 1 is 0.3 while test 2 is 0.6, the probability of being admitted is ',num2str(sigmoid(out * theta)*100),'%']);

