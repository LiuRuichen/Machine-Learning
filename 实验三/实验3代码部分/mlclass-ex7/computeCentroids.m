function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X); %m = 300, n = 2

% You need to return the following variables correctly.
centroids = zeros(K, n);  %K为中心点个数,n实际上为维度


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
counts = zeros(1,K);  %储存每一个类的点的个数

for i = 1:m
    counts(idx(i)) = counts(idx(i)) + 1;  %该类的点个数加一
    centroids(idx(i),:) = centroids(idx(i),:) + X(i,:); %该类的点横纵坐标分别相加，循环结束最终得到每一类所有的点各维度之和
end

for i = 1:K
    centroids(i,:) = centroids(i,:) / counts(i);
end

end

