function [A1, A2] = Affinity_by_tSVD(X, lambda)
%myFun - This is a function for low-rank subspace representation based on t-SVD.
% Syntax: [A1, A2] = Affinity_by_tSVD(X, lambda)
%
% Input : X - data with two views, including features and logical labels
%         lambda - the tradeoff parameter used in this function
% Output: A1 - the subspace representation in the feature space 
%         A2 - the subspace representation of the logical labels
%
% Some codes of this function is inspired by the Xie et al.'s work (IJCV 2018) 
% 
% zhengqinghai@stu.xjtu.edu.cn
% 2019/12/30

num_view = length(X);
num_samples = size(X{1},2); % each column of X{1} and X{2} is a instance

for i = 1:num_view
    Z{i} = zeros(num_samples,num_samples);
    W{i} = zeros(num_samples,num_samples);
    G{i} = zeros(num_samples,num_samples);
    E{i} = zeros(size(X{i},1),num_samples);
    Y{i} = zeros(size(X{i},1),num_samples);
end

w = zeros(num_samples*num_samples*num_view,1);
g = zeros(num_samples*num_samples*num_view,1);
dim1 = num_samples;
dim2 = num_samples;
dim3 = num_view;
size_tensor = [num_samples, num_samples, num_view];

conver_threshold = 1e-7;
conver_flag = 0;
curr_iter = 0;
max_iter = 200;
mu = 1e-4;
mu_max = 1e11;
mu_pho = 2;
rho = 1e-4;
rho_max = 1e13;
rho_pho = 2;

while(conver_flag == 0)
    for i = 1:num_view
        Z_tmp = (X{i}'*Y{i} + mu*X{i}'*X{i} - mu*X{i}'*E{i} - W{i})./rho +  G{i};
        Z{i}=(eye(num_samples,num_samples)+ (mu/rho)*X{i}'*X{i})\Z_tmp;
        E_tmp = [X{1}-X{1}*Z{1}+Y{1}/mu;X{2}-X{2}*Z{2}+Y{2}/mu];
        E_stacked = solve_l1l2(E_tmp,lambda/mu);
        E{1} = E_stacked(1:size(X{1},1),:);
        E{2} = E_stacked(size(X{1},1)+1:end,:);
        Y{i} = Y{i} + mu*(X{i}-X{i}*Z{i}-E{i});
    end
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:);
    w = W_tensor(:);
    [g, objV] = wshrinkObj(z + 1/rho*w,1/rho,size_tensor,0,3);
    G_tensor = reshape(g, size_tensor);
    w = w + rho*(z-g);
    history.objval(curr_iter+1)   =  objV;

    conver_flag = 1;
    for i = 1:num_view
        if (norm(X{i}-X{i}*Z{i}-E{i}, inf) > conver_threshold)
            history.norm_X_Z = norm(X{i}-X{i}*Z{i}-E{i},inf);
            conver_flag = 0;
        end

        G{i} = G_tensor(:,:,i);
        W_tensor = reshape(w, size_tensor);
        W{i} = W_tensor(:,:,i);
        if (norm(Z{i}-G{i}, inf)>conver_threshold)
            history.norm_Z_G = norm(Z{i}-G{i}, inf);
            conver_flag = 0;
        end
    end

    if (curr_iter > max_iter)
        conver_threshold = 1;
    end

    curr_iter = curr_iter + 1;
    mu = min(mu_pho*mu, mu_max);
    rho = min(rho_pho*rho, rho_max);
end
A1 = Z{1};
A2 = Z{2};   
end