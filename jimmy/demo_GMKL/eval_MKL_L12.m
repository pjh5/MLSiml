function mklRes = eval_MKL_L12(dataName, options)
%%%%%%%%%%%%%%%
%% Output
% mklRes

%% Input
% dataName: name of dataset
% options: algorithm environment 

% By Haiqin Yang
%%%%%%%%%%%%%%%

% comparison algorithms
algo1 = 'mklsvm';
algo2 = 'GMKL_Level_L12';
%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------
kernelt={'gaussian' 'gaussian' 'poly' 'poly' }; % 
kerneloptionvect={2.^[-3:6], 2.^[-3:6], [1:3] [1:3]};
variablevec={'all' 'single' 'all' 'single'};

verbose = options.verbose;
nbiter=options.nbiter;
ratio= options.ratio;
C = options.C;

load([dataName]);
    
[n, dim] = size(X);

% times: number of trials
% number of training points, and number of test points
[times,ntr]= size(idx_feature);

% the number of iteration
if nbiter > times
    nbiter = times;
end

%% addust the ratio of training data if needed
newTrSize = floor(n*ratio);

if newTrSize < ntr
    idxLabs=idx_feature(:,1:newTrSize);
    idxUnls=idx_feature(:,1+newTrSize:end);
end

for i= 1 : nbiter
    i
    %% construct training/test  
    xapp = X(idxLabs(i,:),:);
    xtest = X(idxUnls(i,:),:);
    yapp = Y(idxLabs(i,:));
    ytest = Y(idxUnls(i,:));
    
    %% normalization
    [xapp,xtest]=normalizemeanstd(xapp,xtest);
    
    % all possible combinations, for iono: 1+33+1+33
    [kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect); 
    
    % unit trace normalization
    [Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
    
    % combine the kernels
    %  K is a 3-D matrix, where K(:,:,i)= i-th Gram matrix 
    K=mklkernel(xapp,InfoKernel,Weight,options);
    size(K)
    %------------------------------------------------------------------
    % Compare the algorithms
% % % %  %% 
%     For Simple MKL, you may need download the corresponding toolbox  
%     disp('----------    Simple MKL  --------')
%     [SMKL_beta, SMKL_w, SMKL_bsvm, SMKL_posw, SMKL_History, SMKL_fval, SMKL_status] = feval(algo1,K,yapp,C,options,verbose);
%     timeCost(i,1)=SMKL_History.timecost;
%     % test
%     Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(SMKL_posw,:),SMKL_beta);
%     ypred=Kt*SMKL_w+SMKL_bsvm;
% 
%     bc(i,1)=mean(sign(ypred)==ytest)*100
%     selKer(i,1) = length(find(SMKL_beta>options.numericalprecision));
    
    % the number of evaluations
%     nbeval(i,1) = length(SMKL_History.obj);
% % % % %     
    opt = options;
    opt.v = 0;
    [L2_beta, L2_w, L2_bsvm, L2_posw, L2_fval, L2_History] = feval(algo2,K,yapp,C,opt,verbose);
    timeCost(i,2)=L2_History.timecost;

    Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(L2_posw,:),L2_beta);
    ypred=Kt*L2_w+L2_bsvm;

    bc(i,2)=mean(sign(ypred)==ytest)*100
    selKer(i,2) = length(find(L2_beta>options.numericalprecision));

    % the number of evaluations
    nbeval(i,2) = length(L2_History.obj_t);

%     % %     %% Level of L12
% %     %------------------------------------------------------------------
    disp('----------    Level L12 --------')
    [Sigma,w,bsvm,posw,fval,history] = feval(algo2,K,yapp,C,options,verbose); 
    timeCost(i,3)=history.timecost;
    % test
    Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(posw,:),Sigma);
    ypred=Kt*w+bsvm;

    bc(i,3)=mean(sign(ypred)==ytest)*100
    selKer(i,3) = length(find(Sigma>options.numericalprecision));
    
    % the number of evaluations
    nbeval(i,3) = length(history.obj_t); 
end
nbalgo = 3;

if (nbiter==1)
    t = timeCost
    stdt = zeros(1, nbalgo);
    accMKL = bc
    stdMKL = zeros(1, nbalgo);
    selNumK = selKer
    stdNumK = zeros(1, nbalgo);
    nbSVM = nbeval,1
    stdSVM = zeros(1, nbalgo);
else
    t = mean(timeCost,1)
    stdt = timeCost;
    accMKL =mean(bc,1)
    stdMKL = std(bc,1);
    selNumK = mean(selKer,1)
    stdNumK = std(selKer,1);
    nbSVM = mean(nbeval,1)
    stdSVM = std(nbeval,1);
end

%% 
nbalgo = 3;
nbMessure = 4;
temVec = zeros(nbMessure,2*nbalgo);

for i=1: nbalgo
    temVec(1,2*i-1) = t(i);
    temVec(1,2*i) = stdt(i);
    temVec(2,2*i-1) = accMKL(i);
    temVec(2,2*i) = stdMKL(i);
    temVec(3,2*i-1) = selNumK(i);
    temVec(3,2*i) = stdNumK(i);
    temVec(4,2*i-1) = nbSVM(i);
    temVec(4,2*i) = stdSVM(i);
end

mklRes = temVec; 


