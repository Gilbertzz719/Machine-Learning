function p = GMM(x,alpha,mu,sigma)
p=zeros(1,size(x,2));
for i=1:length(alpha)
    p = p + (alpha(i)*mvnpdf(x',mu(:,i)',sigma(:,:,i)))';
end
end