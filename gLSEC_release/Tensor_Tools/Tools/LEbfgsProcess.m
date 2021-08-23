function [target,gradient] = LEbfgsProcess(weights)
%This function is mainly referred to the work proposed by Xu et al. (GLLE IJCAI 2018)

global   trainFeature;
global   trainLabel;
global   para;
global   CC;

[size_sam,size_X]=size(trainFeature);

modProb =  trainFeature * weights; 

L = sum(sum((modProb - trainLabel).^2)); 
R =  trace(modProb'*CC*modProb);
gradL = 2*trainFeature' * (modProb - trainLabel);
gradR =  trainFeature'*CC*trainFeature*weights + ( trainFeature'*CC*trainFeature)'*weights;

target =( L + para * R );
gradient = (gradL +  para * gradR);

end


