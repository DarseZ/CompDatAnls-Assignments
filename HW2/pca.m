function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

U = zeros(n);
S = zeros(n);

% First compute the covariance matrix. Then, use "svd" function to compute the eigenvectors and eigenvalues of the covariance matrix. 
% Note: When computing the covariance matrix, remember to divide by m (the number of examples).

sigma = X'*X/m;

[U,S,V] = svd(sigma);

% =============================================================
end
