function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);   %centroids数组行数，代表中心点个数

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);   %idx代表每个点距离哪一个点最近，取值为1,2,3,...,K，size(X,1)代表样本点个数

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% X每一行都是一个点，一共300个点
for i = 1:size(X,1)
    dist = realmax;
    for j = 1:K
        tag = norm(X(i,:)-centroids(j,:));  %Euclid范数
        if dist > tag
            key = j;
            dist = tag;
        end
    end
    idx(i) = key;
end

% =============================================================

end

