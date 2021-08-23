function [W,numerical] = GLESC(logicalLabel,features,C_low_rank,lambda)
% Some codes of this function is inspired by the work proposed by GLLE (Xu et al. IJCAI 2018)
% zhengqinghai
% 2019/12/30

[d,n] = size(features);
[l,~] = size(logicalLabel);

[kk,~] = size(C_low_rank);

global   trainFeature;
global   trainLabel;
global   para;
global   CC;

ker  = 'rbf'; 
par  = 1*mean(pdist(features)); 
H = kernelmatrix(ker, par, features, features);
UnitMatrix = ones(size(features,1),1);
trainFeature = [H,UnitMatrix];

CC = (eye(kk)-C_low_rank')*(eye(kk)-C_low_rank);
trainLabel = logicalLabel;
para = lambda;
item=rand(size(trainFeature,2),size(trainLabel,2));
[W,fval] = fminlbfgs(@LEbfgsProcess,item);
numerical = trainFeature*W;

end
