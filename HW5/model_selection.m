load cs.mat
% illustrate the picture
figure; imagesc(img); colormap gray;

%% formulize the under-determined linear system
rng(1130);
A = randn(1300,2500);
y = A*double(img(:)) + 5*randn(1300,1);

K = 10;
numSamples = size(A, 1);
numExperiments = 20;
rng(1130);
indices = crossvalind('Kfold', numSamples, K); % gen indices for cross-validation


%% initialize regularization parameters
lambda_set = logspace(-5, 5, numExperiments);
mse_sum = zeros(1, numExperiments);

%% run cross-validation for ridge
for i = 1:K 
    
    A_test = A(indices == i, :);
    A_train = A(indices ~= i, :);
    y_test = y(indices == i, :);
    y_train = y(indices ~= i, :);
    % training
    b_r = ridge(y_train, A_train, lambda_set);
    % out-of-sample error
    e_temp = repmat(y_test, 1, size(lambda_set, 2)) - A_test * b_r;
    mse_sum = mse_sum + sum(e_temp.^2)./size(y_test, 1);
    
end

mse_mean = mse_sum ./ K; % calculate the mse averaged on all the K folds
figure; % plot mse_mean v.s. lamda_set
semilogx(lambda_set, mse_mean);
optimal = find(min(mse_mean) == mse_mean);
text(lambda_set(optimal), mse_mean(optimal), [num2str(lambda_set(optimal)) ' , ' num2str(mse_mean(optimal))], 'HorizontalAlignment', 'left');


%% run cross-validation for Lasso
[b_l, FitInfo] = lasso(A, y, 'CV', K, 'Lambda', lambda_set);
lassoPlot(b_l, FitInfo, 'PlotType', 'CV');
legend('show')

figure; % plot mse_mean v.s. lamda_set
semilogx(FitInfo.Lambda, FitInfo.MSE);
optimal = find(min(FitInfo.MSE) == FitInfo.MSE);
text(FitInfo.LambdaMinMSE, FitInfo.MSE(optimal), [num2str(FitInfo.LambdaMinMSE) ' , ' num2str(FitInfo.MSE(optimal))], 'HorizontalAlignment', 'left');


%% reconstruct image based on ridge coefficients
[r_min_err, r_idx] = min(mse_mean);
b_r_min = b_r(:,r_idx);
figure; imagesc(reshape(b_r_min,50,50)); colormap gray; 
title('Reconstructed image baed on ridge regression')

%% reconstruct image based on Lasso coefficients
[l_min_err, l_idx] = min(FitInfo.MSE);
b_l_min = b_l(:,l_idx);
figure; imagesc(reshape(b_l_min,50,50)); colormap gray; 
title('Reconstructed image baed on Lasso regression')
