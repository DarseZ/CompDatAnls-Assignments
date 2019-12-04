function [Y,idxNN] = my_isomap(X,K,d)
%{
ISYE6740 HW1, By Chi ZHANG
Question 3 main file.
Copyright 2019 Georgia Institute of Technology. All rights reserved.
Inputs:
X - data matrix; K - num of nearest neighborhoods (not included
itself); d - desired dimension of final embedding
Outputs:
Y - embedded data matrix; idxNN
%}
    % Create a NeighborSearcher object for data X with the 'euclidean'
    % distance metric
    Tree = createns(X, 'NSMethod','kdtree','Distance','euclidean');
    
    idxNN = knnsearch(Tree, X, 'K', K+1);
    idxNN = idxNN(: , 2:end); % remove itself
    
    N = size(X,1); % num of observations
    
    W = inf * ones(N,N);
    A = zeros(N,N);
    
    for i = 1:N
        fprintf('Number %d observation...\n', i);
        for k = 1:K
            W(i, idxNN(i,k)) = norm( X(i,:)-X(idxNN(i,k),:), 2); % Compute pairwise distance based on the KNN result
            A(i, idxNN(i,k)) = 1; A(idxNN(i,k), i) = 1; % Mark the neighbor
        end
    end
    % Use Floyd shortest distance algorithm
    D = floyd_dist( min(W,W') );
    H = eye(N)-ones(N,N)/N;
    [Y, E] = eig(-0.5*H*(D.^2)*H); % Y stores the eigenvectors (every column), E stores the eigenvalues (every diagonal element)
    Y = Y(:, 2:d+1);
end