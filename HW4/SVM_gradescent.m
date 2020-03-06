% SVM (Linear)
% Initialization (Dataset: LinearlySeperableData.csv)
%
data=csvread('LinearlySeperableData.csv');
% 
% * *Number of Data Points*

disp(length(data))
% 
% * *Z-score Normalization*

data(:,1:end-1)=zscore(data(:,1:end-1));
% 
% * *Cross Validation*

[train,test] = holdout(data,80);

% Test set
Xtest=test(:,1:end-1);Ytest=test(:,end);

% Traing set
X=train(:,1:end-1);Y=train(:,end);
% 
% * *Visulaization of data*

figure
hold on
scatter(X(Y==1,1),X(Y==1,2),'+g')
scatter(X(Y==-1,1),X(Y==-1,2),'.r')
xlabel('{x_1}')
ylabel('{x_2}')
legend('Positive Class','Negative Class')
title('Data for classification')
hold off



%%

% Applying Sub Gradient Descent with linear Kernel
%

fm_=[];
for s = [5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
          
     % weights,bias
     [W,bias] = gradescent(X,Y,s);
     
     
     % f~ (Predicted labels)
     f=sign(Xtest*W'+bias);
     
     % confusion matrix
     fm = confusion_mat(Ytest, f);
     fm_ = [fm_; s fm];    
end

% Here we use c_optimal to denote the length of learning rate in my
% training process
[min_fm, indx]=min(fm_(:,2));
c_optimal = fm_(indx,1)

% 
% * _*Weights*_

 % weights,bias
 [W,bias]=gradescent(X,Y,c_optimal);


f=sign(Xtest*W'+bias);

%% 
% * _*Performace Measure*_

[F_measure, Accuracy] = confusion_mat(Ytest,f)
% Slack Variables, ?
% *Remark*: Number of Support Vectors and non-zero ? should be nearly same
%
ft = X*W' + bias;
zeta = max(0 , 1-Y.*ft);
Non_Zero_Zeta = sum(zeta~=0)
% Plotting the Hyperplane
%
figure
hold on

scatter(X(Y==1,1),X(Y==1,2),'b')
scatter(X(Y==-1,1),X(Y==-1,2),'r')
%scatter(Xs(Ys==1,1),Xs(Ys==1,2),'.b')
%scatter(Xs(Ys==-1,1),Xs(Ys==-1,2),'.r')

syms x
fn=vpa((-bias-W(1)*x)/W(2),4);
fplot(fn,'Linewidth',2);
fn1=vpa((1-bias-W(1)*x)/W(2),4);
fplot(fn1,'--');
fn2=vpa((-1-bias-W(1)*x)/W(2),4);
fplot(fn2,'--');

axis([-1 1.2 -1 1])
xlabel('X_1')
ylabel('X_2')
title('Hyperplane in 2D')
legend('+ve class','-ve class','Decision Boundry','support vector (+)','support vector (-)','Location','southeast')
hold off

display('W');
W
display('Optimal Objective function value 0.5*W^2');
0.5*norm(W,2)^2

[x1 x2]=meshgrid(min(X(:,1)):0.005:max(X(:,1)),min(X(:,2)):0.005:max(X(:,2)));
Xn=[reshape(x1,1,size(x1,2)*size(x1,1))' reshape(x2,1,size(x2,2)*size(x2,1))'];

yn=sign(Xn*W'+bias);
Yn=reshape(yn,size(x1));

DecisionBoundry( X,Y,Yn )
title('Decision Boundry due to SVM (linear Kernel)')
