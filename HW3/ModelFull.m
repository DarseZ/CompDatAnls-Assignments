function [err_train, err_test] = ModelFull(train, test)
%You implment this function by assuming a full covariance matrix. 
%err_train is the error rate on the train data
%err_test is the error rate on the test data

% train = zscore(train);
% test = zscore(test);

[~, d] = size(train);

% every row is a sample point; last column is the label, which separates
% the xtrain and ytrain
xtrain = double(train(:, 1: d-1));
ytrain = double(train(:, d));

xtest = double(test(:, 1: d-1));
ytest = double(test(:, d));

% scale the inputs in range (0, 1)
xtrain = double(xtrain./255);
xtest = double(xtest ./255);

% find indices for positive (1) and negative (0) labels in the training
% dataset
pos_idx = find(ytrain == 1);
neg_idx = find(ytrain == 0);

% num of occurence for positive and negative samples
pi_pos = length(pos_idx);
pi_neg = length(neg_idx);

% calculate mean of samples for positive and negative
pos_mean = mean(xtrain(pos_idx, :));
neg_mean = mean(xtrain(neg_idx, :));

all_cov = cov(xtrain);

% eigen decomposition of covariance matrix, assume full matrix (\Sigma_1 = \Sigma_0)
[U,S] = eigs(all_cov,d-1);

% add epsilon to avoid dividing zero
epsilon = 1e-2;
V = U * diag(1./sqrt(diag(S) + epsilon));
S = diag(S);

% avoid inversion of covariance matrix by eigen decomposition
pos_train = -sum(((xtrain - repmat(pos_mean, size(train, 1), 1)) * V).^2, 2)/2  + log(pi_pos);
neg_train = -sum(((xtrain - repmat(neg_mean, size(train, 1), 1)) * V).^2, 2)/2  + log(pi_neg);

pos_test = -sum(((xtest - repmat(pos_mean, size(test, 1), 1)) * V).^2, 2)/2  + log(pi_pos);
neg_test = -sum(((xtest - repmat(neg_mean, size(test, 1), 1)) * V).^2, 2)/2  + log(pi_neg);

ytrain_label = max(sign(pos_train - neg_train), 0);
ytest_label = max(sign(pos_test - neg_test), 0);

err_train = sum(abs(ytrain_label - (ytrain)))/length(ytrain);
err_test = sum(abs(ytest_label - ytest))/length(ytest);

end