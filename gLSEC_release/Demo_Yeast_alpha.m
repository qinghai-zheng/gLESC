clc
clear
addpath(genpath(pwd));

fprintf('******EXPERIMENTS ON Yeast-alpha!******\n');

dataset={'LDL_DataSets\Yeast_alpha'};
T=strcat(dataset(1),'.mat');
load(T{1,1});
labelDistribution = labels;
T=strcat(dataset(1),'_binary.mat');
load(T{1,1});
features = zscore(features);

lambda = 10;
beta = 10;

fprintf('-----Get sample relationships based on t-SVD based approach!\n');
X{1} = features';
X{2} = logicalLabel';

[C1, C2] = Affinity_by_tSVD(X, beta);
C_low_rank = (C1+C2)/2;
fprintf('-----finish calculating relationships\n');

[W, numerical] = GLESC(logicalLabel,features,C_low_rank,lambda);
distribution = (softmax(numerical'))';

Result = zeros(6,1);
Result(1,1) = chebyshev(distribution,labelDistribution);
Result(2,1) = clark(distribution,labelDistribution);
Result(3,1) = canberra(distribution,labelDistribution);
Result(4,1) = kldist(distribution,labelDistribution);
Result(5,1) = cosine(distribution,labelDistribution);
Result(6,1) = intersection(distribution,labelDistribution);

fprintf('--------------------------------------\n');
fprintf('\tchebyshev  :%f\n',Result(1,1));
fprintf('\tclark       :%f\n',Result(2,1));
fprintf('\tcanberra    :%f\n',Result(3,1));
fprintf('\tkldist      :%f\n',Result(4,1));
fprintf('\tcosine      :%f\n',Result(5,1));
fprintf('\tintersection:%f\n',Result(6,1));