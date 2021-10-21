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
centroids = zeros(K, n);  %KΪ���ĵ����,nʵ����Ϊά��


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
counts = zeros(1,K);  %����ÿһ����ĵ�ĸ���

for i = 1:m
    counts(idx(i)) = counts(idx(i)) + 1;  %����ĵ������һ
    centroids(idx(i),:) = centroids(idx(i),:) + X(i,:); %����ĵ��������ֱ���ӣ�ѭ���������յõ�ÿһ�����еĵ��ά��֮��
end

for i = 1:K
    centroids(i,:) = centroids(i,:) / counts(i);
end

end

