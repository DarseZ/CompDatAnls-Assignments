house = readtable('RealEstate.csv');
house.Location = categorical(house.Location);
house.Status = categorical(house.Status);

% get the three types of houses
idx_ss = house.Status == 'Short Sale';
house_ss = house(idx_ss, :);

idx_fc = house.Status == 'Foreclosure';
house_fc = house(idx_fc, :);

idx_rg = house.Status == 'Regular';
house_rg = house(idx_rg, :);

% check column names and sizes
house.Properties.VariableNames
fprintf('height of the total')
height(house)
fprintf('height of the short sale')
height(house_ss)
fprintf('height of the foreclosure')
height(house_fc)
fprintf('height of the regular')
height(house_rg)

%% linear regression
model_ss = fitlm(house_ss, 'Price~Bedrooms+Bathrooms+Size+Price_SQ_Ft');
model_fc = fitlm(house_fc, 'Price~Bedrooms+Bathrooms+Size+Price_SQ_Ft');
model_rg = fitlm(house_rg, 'Price~Bedrooms+Bathrooms+Size+Price_SQ_Ft');

%% variables initialization for regularized regression
X_ss = table2array(house_ss(:, 4:7));
y_ss = table2array(house_ss(:, 3));

X_fc = table2array(house_fc(:, 4:7));
y_fc = table2array(house_fc(:, 3));

X_rg = table2array(house_rg(:, 4:7));
y_rg = table2array(house_rg(:, 3));

K = 10;
n_samples_ss = size(X_ss, 1);
n_samples_fc = size(X_fc, 1);
n_samples_rg = size(X_rg, 1);
n_experiments = 50;
rng(1130);
indices_ss = crossvalind('Kfold', n_samples_ss, K); % gen indices for cross-validation
indices_fc = crossvalind('Kfold', n_samples_fc, K); % gen indices for cross-validation
indices_rg = crossvalind('Kfold', n_samples_rg, K); % gen indices for cross-validation
lambda_set = logspace(-7, 7, n_experiments);

%% run ridge regression for the short sale type
mse_sum = zeros(1, n_experiments);
for i = 1:K
    
    X_test = X_ss(indices_ss == i, :);
    X_train = X_ss(indices_ss ~= i, :);
    y_test = y_ss(indices_ss == i, :);
    y_train = y_ss(indices_ss ~= i, :);
    % training
    r_r = ridge(y_train, X_train, lambda_set, 0);
    b_r = r_r(2:size(r_r,1), :);
    c_r = r_r(1, :);
    % out-of-sample error
    e_temp = repmat(y_test, 1, size(lambda_set, 2)) - X_test * b_r - repmat(c_r, size(X_test,1), 1);
    mse_sum = mse_sum + sum(e_temp.^2)./size(y_test, 1);
    
end

mse_mean = mse_sum ./ K; % calculate the mse averaged on all the K folds
figure; % plot mse_mean v.s. lamda_set
semilogx(lambda_set, mse_mean);
optimal = find(min(mse_mean) == mse_mean);
text(lambda_set(optimal), mse_mean(optimal), [num2str(lambda_set(optimal)) ' , ' num2str(mse_mean(optimal))], 'HorizontalAlignment', 'left');

%% run Lasso regression for the short sale type
[b_l, FitInfo] = lasso(X_ss, y_ss, 'CV', K, 'Lambda', lambda_set, 'Standardize',true);

figure; % plot mse_mean v.s. lamda_set
semilogx(FitInfo.Lambda, FitInfo.MSE);
optimal = find(min(FitInfo.MSE) == FitInfo.MSE);
text(FitInfo.LambdaMinMSE, FitInfo.MSE(optimal), [num2str(FitInfo.LambdaMinMSE) ' , ' num2str(FitInfo.MSE(optimal))], 'HorizontalAlignment', 'left');

%% run ridge regression for the foreclosure type
mse_sum = zeros(1, n_experiments);
for i = 1:K
    
    X_test = X_fc(indices_fc == i, :);
    X_train = X_fc(indices_fc ~= i, :);
    y_test = y_fc(indices_fc == i, :);
    y_train = y_fc(indices_fc ~= i, :);
    % training
    r_r = ridge(y_train, X_train, lambda_set, 0);
    b_r = r_r(2:size(r_r,1), :);
    c_r = r_r(1, :);
    % out-of-sample error
    e_temp = repmat(y_test, 1, size(lambda_set, 2)) - X_test * b_r - repmat(c_r, size(X_test,1), 1);
    mse_sum = mse_sum + sum(e_temp.^2)./size(y_test, 1);
    
end

mse_mean = mse_sum ./ K; % calculate the mse averaged on all the K folds
figure; % plot mse_mean v.s. lamda_set
semilogx(lambda_set, mse_mean);
optimal = find(min(mse_mean) == mse_mean);
text(lambda_set(optimal), mse_mean(optimal), [num2str(lambda_set(optimal)) ' , ' num2str(mse_mean(optimal))], 'HorizontalAlignment', 'left');

%% run Lasso regression for the foreclosure type
[b_l, FitInfo] = lasso(X_fc, y_fc, 'CV', K, 'Lambda', lambda_set, 'Standardize',true);

figure; % plot mse_mean v.s. lamda_set
semilogx(FitInfo.Lambda, FitInfo.MSE);
optimal = find(min(FitInfo.MSE) == FitInfo.MSE);
text(FitInfo.LambdaMinMSE, FitInfo.MSE(optimal), [num2str(FitInfo.LambdaMinMSE) ' , ' num2str(FitInfo.MSE(optimal))], 'HorizontalAlignment', 'left');

%% run ridge regression for the regular type
mse_sum = zeros(1, n_experiments);
for i = 1:K
    
    X_test = X_rg(indices_rg == i, :);
    X_train = X_rg(indices_rg ~= i, :);
    y_test = y_rg(indices_rg == i, :);
    y_train = y_rg(indices_rg ~= i, :);
    % training
    r_r = ridge(y_train, X_train, lambda_set, 0);
    b_r = r_r(2:size(r_r,1), :);
    c_r = r_r(1, :);
    % out-of-sample error
    e_temp = repmat(y_test, 1, size(lambda_set, 2)) - X_test * b_r - repmat(c_r, size(X_test,1), 1);
    mse_sum = mse_sum + sum(e_temp.^2)./size(y_test, 1);
    
end

mse_mean = mse_sum ./ K; % calculate the mse averaged on all the K folds
figure; % plot mse_mean v.s. lamda_set
semilogx(lambda_set, mse_mean);
optimal = find(min(mse_mean) == mse_mean);
text(lambda_set(optimal), mse_mean(optimal), [num2str(lambda_set(optimal)) ' , ' num2str(mse_mean(optimal))], 'HorizontalAlignment', 'left');

%% run Lasso regression for the regular type
[b_l, FitInfo] = lasso(X_rg, y_rg, 'CV', K, 'Lambda', lambda_set, 'Standardize',true);

figure; % plot mse_mean v.s. lamda_set
semilogx(FitInfo.Lambda, FitInfo.MSE);
optimal = find(min(FitInfo.MSE) == FitInfo.MSE);
text(FitInfo.LambdaMinMSE, FitInfo.MSE(optimal), [num2str(FitInfo.LambdaMinMSE) ' , ' num2str(FitInfo.MSE(optimal))], 'HorizontalAlignment', 'left');
