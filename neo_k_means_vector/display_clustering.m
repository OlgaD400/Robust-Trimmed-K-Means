function [] = display_clustering(X,ground_C,fname)
% Written by Joyce Whang (joyce@cs.utexas.edu)
% display clustering result (two clusters with overlap and outliers)

X=X';
ground_C=ground_C';

cluster1 = ( ground_C(1,:)~=0 & ground_C(2,:)==0 );
cluster2 = ( ground_C(1,:)==0 & ground_C(2,:)~=0 );
cluster3 = ( ground_C(1,:)~=0 & ground_C(2,:)~=0 );
nocluster = ( ground_C(1,:)==0 & ground_C(2,:)==0 );


plot(X(1,cluster1),X(2,cluster1),'r*','MarkerSize',12)
hold on
plot(X(1,cluster2),X(2,cluster2),'b*','MarkerSize',12)
hold on
plot(X(1,cluster3),X(2,cluster3),'g*','MarkerSize',12)
hold on
plot(X(1,nocluster),X(2,nocluster),'k*','MarkerSize',12)
hold on

set(gca, 'XLim', [-8 8])
set(gca, 'YLim',[-9 9])
set(gca,'FontSize',15)
set(gca,'LineWidth',0.8)
legend('Cluster 1','Cluster 2','Cluster 1 & 2','Not assigned','Location','NW')
print(gcf,sprintf(fname), '-depsc');  
close

end