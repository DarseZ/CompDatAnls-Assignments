function [ class, centroid ] = mykmedoids( pixels, K)

% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns (R, G, and B).
%
%     K: the number of desired clusters.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. The output 
%     should be a matrix with K rows and 3 columns.
%

    % Initialize values
    [m n] = size(pixels);
    centroid = zeros(K, n);
    totalCost = inf;
    % Randomly reorder the indices of examples
    randidx = randperm(m);
    % Take the first K examples as centroids
    centroid = pixels(randidx(1:K), :);
    class = zeros(m, 1);
    
    
    iter = 0;
    % Run K-Medoids
    while (iter < 80)
        
        iter = iter + 1;
        totalCostOld = totalCost;
        totalCost = 0;
        % Output progress
        fprintf('K-Medoids iteration %d...\n', iter);
        
        % For each example in pixels, assign it to the closest centroid
        for i=1:m  %loop for every sample point
            tempMin=inf;
            for j=1:K    %loop for every current centroid
                temp =sum((centroid(j,:)-pixels(i,:)).^2);
                if temp<tempMin
                    class(i)=j; 
                    tempMin = temp;
                end
            end
        end
        
        % the sparse form of a (m by K) matrix, whose the algebraic summation is (m)
        indicator = sparse(1:m, class, 1, m, K, m);
        tempSum = sum(indicator,1);
        % detect if there is any empty cluster
        ifEmpty = [];
        for i = 1:K
            if tempSum(1,i) == 0
                ifEmpty(size(ifEmpty,2)+1) = i;
            end
        end
        % remove the detected empty cluster, then update related variables
        indicator(:,ifEmpty) = [];
        K = size(indicator, 2);
        centroid(ifEmpty,:) = [];

        % Given the memberships, compute new centroids
        for i = 1:K    % loop for centroids
            clu = [];    %awaiting for incoming points
            for j=1:m
                if class(j) == i
                    clu = [clu; pixels(j,:)];
                end
            end
            numMember = size(clu, 1);
            curWithinDist = sum( sqrt(sum( (repmat(centroid(i,:),numMember,1)-clu).^2, 2  ))); % Euclidean distance
            % curWithinDist = sum(sum( abs( repmat(centroid(i,:),numMember,1)-clu ) ));  % Manhattan distance
            
            % Try to find any possibilities to reduce the cost
            for j = 1:numMember    %loop for sample points inside the current cluster
                tempWithinDist = sum( sqrt(sum( (repmat(clu(j,:),numMember,1)-clu).^2, 2  ))); % Euclidean distance
                % tempWithinDist = sum(sum( abs( repmat(clu(j,:),numMember,1)-clu ) )); % Manhattan distance 
                
                if tempWithinDist < curWithinDist 
                    centroid(i,:) = clu(j,:);
                    curWithinDist =  tempWithinDist;
                end
            end
            totalCost = totalCost + curWithinDist;
        end
        
        if abs(totalCost - totalCostOld)/totalCostOld < 1e-4
            fprintf('Achieved convergence with totalCost %.4f...\n', totalCost);
            break;
        end   
    end
% end of the function
end