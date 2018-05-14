function [train_predict,train_accuracy,test_predict,test_accuracy]
=kkr(trainX,trainY,testX,testY,option)

if ~isfield(option,'N')|| isempty(option.N)
    option.N=100;
end
if ~isfield(option,'bias')|| isempty(option.bias)
    option.bias=false;
end
if ~isfield(option,'link')|| isempty(option.link)
    option.link=true;
end
if ~isfield(option,'ActivationFunction')|| isempty(option.ActivationFunction)
    option.ActivationFunction='radbas';
end
if ~isfield(option,'seed')|| isempty(option.seed)
    option.seed=0;
end
if ~isfield(option,'RandomType')|| isempty(option.RandomType)
    option.RandomType='Uniform';
end
if ~isfield(option,'mode')|| isempty(option.mode)
    option.mode=1;
  
end
if ~isfield(option, 'kernelFlag') || isempty(option.mode)
    option.kernelFlag=false;
end
if ~isfield(option, 'kernel') || isempty(option.kernel)
    option.kernel='RBF_kernel'
end
if ~isfield(option,'Scale')|| isempty(option.Scale)
    option.Scale=1;
   
end
if ~isfield(option,'Scalemode')|| isempty(option.Scalemode)
    option.Scalemode=1;
end
rand('state',option.seed);
randn('state',option.seed);
U_trainY=unique(trainY);
nclass=numel(U_trainY);
trainY_temp=zeros(numel(trainY),nclass);
% 0-1 coding for the target 
for i=1:nclass
         idx= trainY==U_trainY(i);
        
         trainY_temp(idx,i)=1;
end
%  trainY_temp = trainY; % for regression
[Nsample,Nfea]=size(trainX);
N=option.N;
if strcmp(option.RandomType,'Uniform') 
    if option.Scalemode==3
         Weight= option.Scale*(rand(Nfea,N)*2-1);
         Bias= option.Scale*rand(1,N);
      %   fprintf('linearly scale the range of uniform distribution to %d\n',  option.Scale);
    else
         Weight=rand(Nfea,N)*2-1;
          Bias=rand(1,N);
    end
else if strcmp(option.RandomType,'Gaussian')
          Weight=randn(Nfea,N);
          Bias=randn(1,N);  
    else
        error('only Gaussian and Uniform are supported')
    end
end
Bias_train=repmat(Bias,Nsample,1);
H=trainX*Weight+Bias_train;

H = trainX;



H(isnan(H))=0;

%p_value = 50;
if option.kernelFlag
    Htr = H;
    H = kernel_matrix(H,option.kernel, option.p_value);
end


if option.mode==2
    
beta=pinv(H)*trainY_temp;
else if option.mode==1
        
    if ~isfield(option,'C')||isempty(option.C)
        option.C=0.1;
    end
    C=option.C;
    if N<Nsample
     beta=(eye(size(H,2))/C+H' * H) \ H'*trainY_temp;
    else
     beta=H'*((eye(size(H,1))/C+H* H') \ trainY_temp); 
    end
    else
      error('Unsupport mode, 
        only Regularized least square and Moore-Penrose pseudoinverse are allowed. ')  
    end
end
trainY_temp=H*beta;
Y_temp=zeros(Nsample,1);
% decode the target
for i=1:Nsample
    [maxvalue,idx]=max(trainY_temp(i,:));
    Y_temp(i)=U_trainY(idx);
 end

Bias_test=repmat(Bias,numel(testY),1);
H_test=testX*Weight+Bias_test;
H_test = testX;


 

H_test(isnan(H_test))=0;
if option.kernelFlag
    H_test = kernel_matrix(Htr,option.kernel, option.p_value, H_test);
    H_test = H_test';
end
testY_temp=H_test*beta;
Yt_temp=zeros(numel(testY),1);

for i=1:numel(testY)
    [maxvalue,idx]=max(testY_temp(i,:));
    Yt_temp(i)=U_trainY(idx);
end

% train_predict = trainY_temp;
train_predict = Y_temp; 
% test_predict = testY_temp;
test_predict = Yt_temp;
train_accuracy=length(find(Y_temp==trainY))/Nsample;

test_accuracy=length(find(Yt_temp==testY))/numel(testY);


end

function [Output,k,b]=Scale_feature(Input,Saturating_threshold,ratio)
Min_value=min(min(Input));
Max_value=max(max(Input));
min_value=Saturating_threshold(1)*ratio;
max_value=Saturating_threshold(2)*ratio;
k=(max_value-min_value)/(Max_value-Min_value);
b=(min_value*Max_value-Min_value*max_value)/(Max_value-Min_value);
Output=Input.*k+b;
end

function [Output,k,b]=Scale_feature_separately(Input,Saturating_threshold,ratio)
nNeurons=size(Input,2);
k=zeros(1,nNeurons);
b=zeros(1,nNeurons);
Output=zeros(size(Input));
min_value=Saturating_threshold(1)*ratio;
max_value=Saturating_threshold(2)*ratio;
for i=1:nNeurons
Min_value=min(Input(:,i));
Max_value=max(Input(:,i));
k(i)=(max_value-min_value)/(Max_value-Min_value);
b(i)=(min_value*Max_value-Min_value*max_value)/(Max_value-Min_value);
Output(:,i)=Input(:,i).*k(i)+b(i);
end
end

    
function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);


if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end


end