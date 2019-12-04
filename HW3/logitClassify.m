function [err_train, err_test] = logitClassify(train, test, eta)
%You implment this function by assuming a full covariance matrix. 
%err_train is the error rate on the train data
%err_test is the error rate on the test data

% train = zscore(train);
% test = zscore(test);


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

% distance matrix
% Initialize pairwise distance matrix
m_train = size(xtrain,1);
m_test = size(xtest,1);

% random initialization of theta
theta = rand(d-1,1);
theta_old = zeros(d-1,1);

% specify stoping threshold
epsilon = 0.01;

while(norm (theta-theta_old) > epsilon)
    %disp (norm (theta-theta_old));
    iter_sum = zeros(d-1,1);
    for i = 1: m_train
        iter_sum = iter_sum + (ytrain(i,:) - 1/(1+exp(-xtrain(i,:)*theta)))* xtrain(i,:)';
    end 
    theta_old = theta;
    theta = theta_old + eta * iter_sum;     
end    
    
% classify on training set
ytrain_label = (1./ (exp(-xtrain * theta)+1)) > 0.5;

% classify on test set
ytest_label = (1./ (exp(-xtest * theta)+1)) > 0.5;

err_train = sum(abs(ytrain_label - (ytrain)))/length(ytrain);
err_test = sum(abs(ytest_label - ytest))/length(ytest);

end