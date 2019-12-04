function [err_train, err_test] = ModelSpherical(train, test)
%You implment this function by assuming a spherical covariance matrix. 
%err_train is the error rate on the train data
%err_test is the error rate on the test data

[~, d] = size(train);

% last column of train is label, separate xtrain and xtrain
xtrain = double(train(:, 1: d-1));
ytrain = double(train(:, d));

% last column of test is label, separate xtrain and xtrain
xtest = double(test(:, 1: d-1));
ytest = double(test(:, d));

% scale the inputs in range (0,1)
xtrain = double(xtrain./255);
xtest = double(xtest ./255);

% find indices for positive (+1) and negative (0) labels in the training
% dataset
pos_idx = find(ytrain == 1);
neg_idx = find(ytrain == 0);

% number of occurence for positive and negative in the training set
pi_pos = length(pos_idx);
pi_neg = length(neg_idx);

% calculate mean of samples for positive and negative
pos_mean = mean(xtrain(pos_idx, :));
neg_mean = mean(xtrain(neg_idx, :));

% all features have the fixed variance of sigma square
% zero covariance between variates
sigma = 1;
all_cov = diag(eye(d-1).*sigma);

% add epsilon to avoid dividing zero
epsilon = 1e-2;
V = diag(1./sqrt(all_cov + epsilon));   

pos_train = -sum(((xtrain - repmat(pos_mean, size(train, 1), 1)) * V).^2, 2)/2  + log(pi_pos);
neg_train = -sum(((xtrain - repmat(neg_mean, size(train, 1), 1)) * V).^2, 2)/2  + log(pi_neg);

pos_test = -sum(((xtest - repmat(pos_mean, size(test, 1), 1)) * V).^2, 2)/2  + log(pi_pos);
neg_test = -sum(((xtest - repmat(neg_mean, size(test, 1), 1)) * V).^2, 2)/2  + log(pi_neg);

ytrain_label = max(sign(pos_train - neg_train), 0);
ytest_label = max(sign(pos_test - neg_test), 0);

err_train = sum(abs(ytrain_label - (ytrain)))/length(ytrain);
err_test = sum(abs(ytest_label - ytest))/length(ytest);

end