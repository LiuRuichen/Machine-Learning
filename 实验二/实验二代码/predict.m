function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');  %5000*401 * 401*25 = 5000*25
h2 = sigmoid([ones(m, 1) h1] * Theta2'); %5000*26 * 26*10 = 5000*10
[dummy, p] = max(h2, [], 2);  

% =========================================================================


end
