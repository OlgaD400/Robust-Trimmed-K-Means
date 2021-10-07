function [U,J] = neo_kmeans_copy(X,k,alpha,beta,initU)
%% NEO-K-Means
% Written by Joyce Whang (joyce@cs.utexas.edu)
% Non-exhaustive, Overlapping k-means
% J. J. Whang, I. S. Dhillon, and D. F. Gleich
% SIAM International Conference on Data Mining (SDM), 2015

N = size(X,1);   % no. of data points
dim = size(X,2); % dimension
U = initU;       % initial U

%% Overlapping and non-exhaustive clustering
t=0;
t_max=100;
alphaN = round(alpha*N);
betaN = round(beta*N);
J=Inf;
oldJ=0;
epsilon=0;
J_track=[];
D=zeros(N,k);   % distance matrix (nodes by clusters)
M=zeros(dim,k); % cluster mean (dimension by clusters)

while abs(oldJ-J)>epsilon && t<=t_max
    oldJ=J;
    J=0;
    %% Compute cluster means
    for j=1:k
        ind = logical(U(:,j));
        members = nnz(ind);
        
        if members~=0
            M(:,j) = mean(X(ind,:));
        else
            M(:,j) = rand(dim,1);
        end
    end
    
    %% Compute distance
    for j=1:k
        diff = X - repmat(M(:,j)',N,1);
        %D(:,j) = sqrt(sum(diff.^2,2));
        D(:,j) = sum(diff.^2,2);
    end
    
    %% Make (N-betaN) assignments
    [dist,ind] = min(D,[],2);
    dnk = [dist (1:N)' ind]; % N by 3 matrix (distance node min_k)
    sorted_dnk = sortrows(dnk,1);
    sorted_d = sorted_dnk(:,1);
    sorted_n = sorted_dnk(:,2);
    sorted_k = sorted_dnk(:,3);
    numAssign = N-betaN;
    J = J + sum(sorted_d(1:numAssign));
    U = sparse(sorted_n(1:numAssign), sorted_k(1:numAssign), ones(numAssign,1), N, k);
    temp=[sorted_n(1:numAssign), sorted_k(1:numAssign)];
    for p=1:numAssign
        D(temp(p,1),temp(p,2))=Inf;
    end

    %% Make (alphaN + betaN) assignments
    n=0;
    while n < (alphaN + betaN)
        min_d = min(min(D));
        J = J+min_d;
        [i_star, j_star] = find(D==min_d);
        U(i_star(1),j_star(1)) = 1; % assign
        D(i_star(1),j_star(1)) = Inf; % don't consider this assignment again
        n=n+1;
    end
    t=t+1;
    J_track=[J_track; J];
    fprintf('***** iteration: %d, objective: %6.6f\n',t,J);
end
fprintf('***** No. of iterations done: %d\n',t);


%% display results
U = full(U);

fprintf('***** Total no. of data points: %d \n',N);
fprintf('***** alpha: %3.3f, alphaN: %d \n',alpha,alphaN);
fprintf('***** beta: %3.3f, betaN: %d \n',beta,betaN);


end