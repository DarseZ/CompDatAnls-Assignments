function [train, test] = SplitData(p)
% SplitData helps you to randomly split the data. 
% p is the proportion of the train set.
% usage : [train, test] = SplitData(0.8); will use 80-percent of the data
% to train and 20-percent to test.

% mydata = dlmread('Iris.txt');
load('usps-2cls.mat');
N = size(mydata,1);

idx = randperm(N);

num_train_samples = floor(N * p);

train = mydata(idx(1:num_train_samples), :);

test = mydata(idx(num_train_samples + 1:end), :);

end