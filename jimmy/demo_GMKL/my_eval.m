function mklRes = my_eval(data_option, options)
% require, data_options.X, data_options.y, data_options.kernel_options, 

algo1 = 'mklsvm';
algo2 = 'GMKL_Level_L12';
%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------

verbose = options.verbose;
nbiter=options.nbiter;
ratio= options.ratio;
C = options.C;
X = data_option.X;
y = data_option.y;
num_exps = data_option.cv;


    
[n, dim] = size(X);
cv_indices = crossvalind('Kfold', n, num_exps);

    function [train_ind, cv_ind] = cv_split(iter)
        train_ind = find(cv_indices~=iter);
        cv_ind = find(cv_indices==iter);
    end
% times: number of trials
% number of training points, and number of test points
% for i = 1: num_exps
for i = 1: 1
    [train_ind, cv_ind] = cv_split(i);
    xapp = X(train_ind,:);
    xtest = X(cv_ind,:);
    yapp = y(train_ind);
    ytest = y(cv_ind);
     [xapp,xtest]=normalizemeanstd(xapp,xtest);
     [kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(data_option.indices,dim,data_option.kernel_type,data_option.kernel_param); 
     
     % unit trace normalization
    [Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
    
     K=mklkernel(xapp,InfoKernel,Weight,options);
     size(K)
     
     opt = options;
    opt.v = 0;
    [L2_beta, L2_w, L2_bsvm, L2_posw, L2_fval, L2_History] = feval(algo2,K,yapp,C,opt,verbose);
    timeCost(i,2)=L2_History.timecost;

    Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(L2_posw,:),L2_beta);
    disp('weight used for combination is ')
    L2_beta = L2_beta
    
    ypred=Kt*L2_w+L2_bsvm;

    bc(i,2)=mean(sign(ypred)==ytest)*100
    selKer(i,2) = length(find(L2_beta>options.numericalprecision));
    
     % the number of evaluations
    nbeval(i,2) = length(L2_History.obj_t);
    
    %     % %     %% Level of L12
    % %     %------------------------------------------------------------------
%     disp('----------    Level L12 --------')
%     [Sigma,w,bsvm,posw,fval,history] = feval(algo2,K,yapp,C,options,verbose); 
%     timeCost(i,3)=history.timecost;
%     % test
%     Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(posw,:),Sigma);
%     ypred=Kt*w+bsvm;
%     disp('weight used for combination is ')
%     Sigma = Sigma
% 
%     bc(i,3)=mean(sign(ypred)==ytest)*100
%     selKer(i,3) = length(find(Sigma>options.numericalprecision));
%     
%     % the number of evaluations
%     nbeval(i,3) = length(history.obj_t); 
     
end
nbalgo = 2;
nbiter = 1;
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
nbalgo = 2;
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



end