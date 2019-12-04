function [err_train, err_test] = knnClassify(train, test, k)
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


distance_matrix_train = zeros(m_train, m_train);

    for i = 1:m_train
        for j = i+1:m_train
            distance_matrix_train(i,j) = norm(xtrain(i,:)-xtrain(j,:));
        end
    end 
    distance_matrix_train = distance_matrix_train' + distance_matrix_train;
    
    for i =1:m_train
        sorted_distances = sort(distance_matrix_train(i,:));
        knn_indices = find(ismember(distance_matrix_train(i,:), sorted_distances(2:k+1)));
        ytrain_label(i)= mode(ytrain(knn_indices,:));
    end


distance_matrix_test = zeros(m_test, m_train);

    for i = 1:m_test
        for j = 1:m_train
            distance_matrix_test(i,j) = norm(xtest(i,:)-xtrain(j,:));
        end
    end 

    for i =1:m_test
        sorted_distances = sort(distance_matrix_test(i,:));
        knn_indices = find(ismember(distance_matrix_test(i,:), sorted_distances(1:k)));
        ytest_label(i)= mode(ytrain(knn_indices,:));
    end

ytrain_label = ytrain_label';
ytest_label = ytest_label';

err_train = sum(abs(ytrain_label - (ytrain)))/length(ytrain);
err_test = sum(abs(ytest_label - ytest))/length(ytest);

end