clear,clc,close all
%%Question 3
regweight=1e-2;
%Step 1 Generate samples from a 4-component GMM
% alpha_true=[0.3,0.1,0.2,0.4];
% mu_true=[-1,3,12,-8;9,-5,-8,6];
% sigma_true(:,:,1) = [7 -3;-3 8]/6; sigma_true(:,:,2) = [8 2;2 4]/5;
% sigma_true(:,:,3) = [8 -2;-2 4]/6; sigma_true(:,:,4) = [4 1;1 3]/2;
alpha_true=[0.2,0.3,0.23,0.27]; mu_true=[-5 -1 8 3;5 1 3 -9];
sigma_true(:,:,1)=[3,2;2,15]/2; sigma_true(:,:,2)=[9,3;3,3]/2;
sigma_true(:,:,3)=[7,4;4,16]/2; sigma_true(:,:,4)=[5,3;3,2]/2;
% alpha_true=[0.2,0.3,0.23,0.27]; mu_true=[-1 4 5 -5;5 -7 7 -9];
% sigma_true(:,:,1)=[3,2;2,15]/2; sigma_true(:,:,2)=[9,3;3,3]/2;
% sigma_true(:,:,3)=[7,4;4,16]/2; sigma_true(:,:,4)=[5,3;3,2]/2;
N=1000;
x=data_generate(N,alpha_true,mu_true,sigma_true);
figure(1)
plot(x(1,:),x(2,:),'.b')
xlabel('x1');ylabel('x2')
M=100;
B=10;
C=6;
converge=0;
I=150;
delta=0.001;
exp_result=zeros(1,M);
for m=1:M
    %Step 1: Data generation
    N=1000;
    x=data_generate(N,alpha_true,mu_true,sigma_true);
    [d,~]=size(x);
    ratio_set=linspace(0.4,0.9,6);
    for b=1:B
        x=x(:,randperm(N));
        r=randperm(4);
        ratio=ratio_set(r(1));
        n=floor(ratio*N);
        train=x(:,1:n);
        test=x(:,(n+1):N);
        for c=1:C
            alpha=ones(1,c)/c;
            rs=randperm(N);
            mu=x(:,rs(1:c));
            [~,labels] = min(pdist2(mu',train'),[],1);
            for l=1:c
                label=(find(labels==l));
                if length(label)>0
                    sigma(:,:,l)=cov(train(:,label)')+regweight*eye(d,d);
                else
                    sigma(:,:,l)=[1,0.5;0.5,1];
                end
            end
            llh=sum(log(GMM(train,alpha,mu,sigma)));
            temp=zeros(c,n);
            for k=1:I
            %while ~converge
                for i=1:c
                    temp(i,:)=alpha(i).*(mvnpdf(train',mu(:,i)',sigma(:,:,i))');
                end
                normalize=temp./sum(temp,1);
                new_alpha=mean(normalize,2);
                distribution=normalize./(sum(normalize,2)+1);
                new_mu=train*distribution';
                for i=1:c
                    v=train-repmat(new_mu(:,i),1,n);
                    u=repmat(distribution(i,:),d,1).*v;
                    new_sigma(:,:,i)=u*v'+regweight*eye(d,d); 
                end
                new_llh=sum(log(GMM(test,new_alpha,new_mu,new_sigma)));
                d_alpha=sum(abs(new_alpha-alpha'));
                d_mu=sum(sum(abs(new_mu-mu)));
                d_sigma=sum(sum(abs(abs(new_sigma-sigma))));
                d_llh=abs(llh-new_llh);
                alpha=new_alpha; mu = new_mu; sigma=new_sigma;
                %converge = (d_llh<delta);
                converge = ((d_alpha+d_mu+d_sigma)<delta);
                llh=new_llh;
            end
%             llh_result(c,(m-1)*B+b)=llh;
            llh_result(c,b)=llh;
        end
%         [~,ind]=max(llh_result(:,(m-1)*B+b));
%         selection((m-1)*B+b)=ind;
        [~,ind]=max(llh_result(:,b));
        selection_1(b)=ind;
    end
    count=zeros(1,6);
    for i=1:6
        count(i)=count(i)+length(find(selection_1==i));
    end
    [~,a]=max(count);
    exp_result(1,m)=a;
end
llh_est=mean(llh_result,2);
count=zeros(1,6);
% for i=1:6
%     count(i)=count(i)+length(find(selection==i));
% end
figure(2)
xbins=1:6;
% hist(selection,xbins);
hist(exp_result,xbins);
title('Number of selction for each number of components for 10 samples')
xlabel('Number of Components')


