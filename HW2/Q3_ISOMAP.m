%% =============== Step 1: Loading and Visualizing Face Data =============
%{
ISYE6740 HW1, By Chi ZHANG
Question 3 main file.
Dependent functions:
    my_isomap.m
    floyad_dist.m
Copyright 2019 Georgia Institute of Technology. All rights reserved.
%}

%  Load Face dataset
load('isomap.mat');
%  Make every face stored in a row
images = images';
%  Display the first 144 faces in the dataset
displayData(images(1:144, :), 64);
title('The first 144 faces in the dataset')

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Step 2: Feature Normalization ==============
%  Before running ISOMAP, it is important to first normalize X by subtracting 
%  the mean value from each feature
[images_norm, mu, sigma] = featureNormalize(images);
%  Visualize the mean face found
figure;
displayData(mu, 64);
title('The mean face')
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Step 3: Run Isomap ==============

[Y, idxNN] = my_isomap(images_norm, 20, 2);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Step 4: Do Visualization ============= 
figure;
scatter(Y(:,1),Y(:,2));
title('Embedded data structure with num 20 of nearest neighborhoods')
hold
scatter(Y(8,1),Y(8,2),'filled','r');
scatter(Y(11,1),Y(11,2),'filled','r');
scatter(Y(21,1),Y(21,2),'filled','r');

fprintf('Program paused. Press enter to continue.\n');
pause;