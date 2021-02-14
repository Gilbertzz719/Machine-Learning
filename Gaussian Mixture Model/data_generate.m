function data=data_generate(N,alpha,mu,sigma)
pc=[0,cumsum(alpha)]; r=rand(1,N); 
data=zeros(size(mu,1),N);
for i=1:length(alpha)
    index=(find(pc(i)<r&r<=pc(i+1))); 
    data(:,index)=mvnrnd(mu(:,i),sigma(:,:,i),length(index))';
end