%% =============== Step 1: Loading and Visualizing Face Data =============
%{
ISYE6740 HW1, By Chi ZHANG
Question 1 main file.
Copyright 2019 Georgia Institute of Technology. All rights reserved.

Notes:
Make sure the current working path only include the specified training
image set(.gif file), not any irrelavent .gif file. Besides, in step5, use
absolute path to load your testing image for face recognition.
%}

fprintf('\nLoading face dataset.Ensure the current folder has only the trained .gif images!!!\n\n');
list = dir('*.gif');
% Initialize the data matrix, every row is an image
X = [];
% Downsampled img width
imgW = 0;
% Store all the subject14XXX.gif files into the working path, except the subject14.test.gif
for k = 1:size(list,1)
    temp = list(k).name; 
    img = imread(temp);
    % Downsampling the image to reduce the resolution
    img = img(1:4:end,1:4:end);
    imgW = size(img,2);
    % Transform the img matrix to a row vector for every single image
    pixels = zeros(1,size(img,1)*size(img,2));
	for i=1:size(img,1)
		for j=1:size(img,2)
			pixels((j-1)*size(img,1)+i) = img(i,j);
		end
    end
    % Add it to the data matrix
    X = [X ; pixels];
end

%  Display the faces in the dataset
displayData(X, imgW);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Step 2: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 6 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Visualize the mean face found
displayData(mu, imgW);

figure;
%  Visualize the top 6 eigenvectors found
displayData(U(:, 1:6)', imgW);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Step 3: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 6;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% ==== Step 4: Visualization of Faces after PCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 6;
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm, imgW);
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec, imgW);
title('Recovered faces');
axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==== Step 5: Test Face Recognition by the 1st principal component  ==
%  Project test images using the top 1 eigenvector and visualize the recovered image only using 1 dimension information
%  Compare to the original input, which is also displayed
fprintf('\nDimension reduction for test face; Then, Visualizing the projected test face.\n\n');

% Load the test image by absolute path.
% Two test images:
% 'E:\Richard-Production\ISYE6740_Codes\yalefaces\subject01.gif','E:\Richard-Production\ISYE6740_Codes\yalefaces\subject14.test.gif'
testImg = imread('E:\Richard-Production\ISYE6740_Codes\yalefaces\subject14.test.gif');
% Downsampling the image to reduce the resolution
testImg = testImg(1:4:end,1:4:end);
imgW = size(testImg,2);
pixels = zeros(1,size(testImg,1)*size(testImg,2));
for i=1:size(testImg,1)
    for j=1:size(testImg,2)
        pixels((j-1)*size(testImg,1)+i) = testImg(i,j);
    end
end
[test_norm, test_mu, test_sigma] = featureNormalize(pixels);
testZ = projectData(test_norm, U, 1); % Project testing image to the 1st principal component of training data.

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(testZ));

fprintf('\nVisualizing the projected (reduced dimension) test face.\n\n');

test_rec  = recoverData(testZ, U, 1); % Recover projected information to high-dimensional space.

% Display normalized data
subplot(1, 2, 1);
displayData(test_norm, imgW);
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(test_rec, imgW);
title('Recovered faces');
axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;