%This is the main routine for homework 4
%You are asked to plugin your implementation for the funciton ModelFull,
%ModelDiagonal, and ModelSpherical

%Repeat the experiments for 100 times
N = 100;

err_Full = zeros(N,2);
err_Diagonal = zeros(N,2);
err_Spherical = zeros(N,2);


%Let p change from 0.1, 0.2, 0.5, 0.8, 0.9 to compare the performance of each classifier
p = 0.1;

for i = 1 : N
	
	[train, test] = SplitData(p);
	
	[err_train, err_test] = ModelFull(train, test);
	err_Full(i,:) = [err_train, err_test];
	
	[err_train, err_test] = ModelDiagonal(train, test);
	err_Diagonal(i,:) = [err_train, err_test];
	
	[err_train, err_test] = ModelSpherical(train, test);
	err_Spherical(i,:) = [err_train, err_test];
	
	
end

mean_err_Full = mean(err_Full);
mean_err_Diagonal = mean(err_Diagonal);
mean_err_Spherical = mean(err_Spherical);

fprintf('err_Full : %g, %g\n', mean_err_Full(1), mean_err_Full(2));
fprintf('err_Diagonal : %g, %g\n', mean_err_Diagonal(1), mean_err_Diagonal(2));
fprintf('err_Spherical : %g, %g\n', mean_err_Spherical(1), mean_err_Spherical(2));

%%
% THE KNN CLASSIFIER TAKES LONG TIME TO COMPLETE DUE TO THE LARGE NUMBER OF TEST CASES SPECIFIED IN THE QUESTION
% Repeat the experiments for 100 times
N = 100;
% Let p change from 0.1, 0.2, 0.5, 0.8, 0.9 to compare the performance of each classifier
p = 0.1;
err_Knn = zeros(N,2);

% k specifies number of nearest neighbors in knn classifier
k = [5, 10, 15, 30];

for z = 1:4

    for i = 1 : N
            
        % split the training/test data by p:(1-p)
        [train, test] = SplitData(p);
            
        % knn classifier
        [err_train, err_test] = knnClassify(train, test, k(z));
        err_Knn(i,:) = [err_train, err_test];
    end
    % calcualte mean error for N iterations
    mean_err_Knn = mean(err_Knn);
    fprintf('k=%g, p=%g, err_Knn : %g, %g\n', k(z), p,mean_err_Knn(1), mean_err_Knn(2));

end

%%
N = 100;

err_Logit = zeros(N,2);

mean_err_Logit_list = [];

%Let p change from 0.1, 0.2, 0.5, 0.8, 0.9 to compare the performance of each classifier
p = [0.1, 0.2, 0.5, 0.8, 0.9];
eta = 0.01;

for j = p
    
    tic;
    for i = 1 : N
        % split the training/test data by p:(1-p)
        [train, test] = SplitData(j);
        
        % logistics regression classifier
        [err_train, err_test] = logitClassify(train, test, eta);
        err_Logit(i,:) = [err_train, err_test];
        
    end
    
    % calcualte mean error for N iterations
    mean_err_Logit = mean(err_Logit);
    mean_err_Logit_list = vertcat(mean_err_Logit_list, mean_err_Logit);
    casetime = toc;
    
    fprintf('p=%g, err_Full : %g, %g\n', j, mean_err_Logit(1), mean_err_Logit(2));
    fprintf('Time for completing test case p=%g is : %g sec\n', j ,casetime);
end