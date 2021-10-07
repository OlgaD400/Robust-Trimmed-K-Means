function [alpha,beta] = estimate_alpha_beta(X,C,alpha_delta,beta_delta)
% Written by Joyce Whang (joyce@cs.utexas.edu)
% X: no. of points by dimension
% C: k cluster centers (k by dim)
% alpha_delta: (1) between -1 and 3.5 (the first strategy) 
%              (2) alpha_delta='' (the second strategy) 
% beta_delta: 3 or 6

N = size(X,1);
k = size(C,1);
D = zeros(N,k); % distance matrix (nodes by clusters)

%% Compute distance
for j=1:k
    diff = X - repmat(C(j,:),N,1);
    D(:,j) = sqrt(sum(diff.^2,2));
end
[dist,ind] = min(D,[],2);

%% Estimate beta
betaN = nnz( dist > (mean(dist)+beta_delta*std(dist)) );
beta = betaN/N;
disp(beta)

%% Estimate alpha
if ischar(alpha_delta)
    rdist = D./repmat(sum(D,2),1,k);
    alpha=nnz(rdist<(1/(k+1)))/N;
else
    ovlap = 0;
    for j=1:k
        if nnz(ind==j)>0
            cdist = dist(ind==j);
            threshold = mean(cdist)+alpha_delta*std(cdist);
            v = ( D(ind~=j,j) <= threshold );
            ovlap = ovlap + nnz(v);
        end
    end
    alpha = ovlap/N;
    disp(alpha)
end


end