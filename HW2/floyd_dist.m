function D = floyd_dist(D_0)
% The Floyd shortest distance algorithm is indeed a dynamic-programming
% approach. Update the shortest btw any two points i and j, if find existing the
% third point could be used as a pivotal k s.t. Dist(i,k) + Dist(k,j) < Dist(i,j)
    D = D_0;
    N = size(D,1);
    
    for k = 1:N
        % disp(k);
        D = min( D, repmat(D(:,k),[1,N])+repmat(D(k,:),[N,1]) );
    end
end