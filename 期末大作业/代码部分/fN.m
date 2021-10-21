function [X_norm mu] = fN(X)
mu = mean(X); 
X_norm = bsxfun(@minus, X, mu);  
end

