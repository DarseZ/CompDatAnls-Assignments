function [w,b] = gradescent( X,Y,s )

% s is the constant multiple of each iterations
% The parameter here is b and w

N=size(X,1);
% Tolerence (for convergence)
norm1=Inf;
tol=10e-12;

total_iterations=0;

w2 = zeros(1,2);
b = 0;

while norm1>tol
    
    total_iterations=total_iterations+1;
    w1 = w2;
    ttt = 0;
    for i=1:N
        if 1 - (w1*X(i,:)'+b)*Y(i)>0
            w2 = w1-(-s*X(i,:)*Y(i));
            b = b-(-s*Y(i));
            ttt = 1;
            break;
        end     
    end
    if ttt==0
        w2 = w1-2*s*w1;
    end
    norm1=norm(w2-w1);
    display(norm1);
    if total_iterations==10000
        break;
    end
end

display(total_iterations);

w = w2;

end

