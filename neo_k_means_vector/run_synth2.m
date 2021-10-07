%% NEO-K-Means
% Written by Joyce Whang (joyce@cs.utexas.edu)
% Non-exhaustive, Overlapping k-means
% J. J. Whang, I. S. Dhillon, and D. F. Gleich
% SIAM International Conference on Data Mining (SDM), 2015

clear all;
addpath('./neo_kmeans');

%% load data
load synth2.mat

%% parameters
N = size(X,1); % No. of data points
k = 2;         % No. of clusters

%% initialize
[IDX,C] = kmeans(X,k);
initU = zeros(size(X,1),k);
for kk=1:k
    initU(:,kk) = (IDX==kk);
end
    
%% set alpha, beta
% 1. alpha/beta can be given by a user
%    alpha = 0.1;
%    beta = 0.005;
% 2. alpha/beta can be estimated (see the paper for details).
%    hyperparameters alpha_delta/beta_delta can be set as follows:
%    alpha_delta: (1) between -1 and 3.5 (the first strategy)
%                 (2) alpha_delta='' (the second strategy)
%    beta_delta: 3 or 6
alpha_delta = 1.9; 
beta_delta = 6;    
[alpha,beta] = estimate_alpha_beta(X,C,alpha_delta,beta_delta);

%% run neo-kmeans
[U,J] = neo_kmeans(X,k,alpha,beta,initU);

%% visualize the clustering
display_clustering(X,U,'neo_kmeans_output');
display_clustering(X,ground_C,'ground_truth');