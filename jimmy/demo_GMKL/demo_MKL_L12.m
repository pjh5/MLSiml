%% Settings
%% Currently we only have binary classification
%% Multi-class and regression will be added soon.
options.algo='svmclass'; % Choice of algorithm in mklsvm can be either
                         % 'svmclass' or 'svmreg'
%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation= 1; % use variation of weights for stopping criterion 
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
options.stopdualitygap = 1; % set to 1 for using duality gap for stopping criterion
options.stopapproxgap = 1;

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildifftheta= 1e-3;        % stopping criterion for weight variation 
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap= 1e-2;       % stopping criterion for duality gap
options.seuilboundgap= 1e-2;       % stopping criterion for level set gap

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                   % variable in the reduced gradient method 
options.nbitermax= 500;             % maximal number of iteration  
options.seuil=0;                   % forcing to zero weights lower than this 
options.seuilitermax=10;           % value, for iterations lower than this one 
options.numericalprecision = 1e-5; % numberical precision
options.miniter=0;                 % minimal number of iterations 
options.verbosesvm=0;              % verbosity of inner svm algorithm 
options.efficientkernel = 0;         % use efficient storage of kernels 

% MMS-MKL SVM Level
options.lambda = 0.90;
options.lambdaFinal = 0.99;

options.verbose= 0;
options.nbiter=  2;
options.ratio= 0.5;

options.C = [100];

options.v = 0.5;

% We have implemented four solvers: CVX, Yalmip, 'Mosek', 'Mosek_Conic'
% You need to add the path or set up correctly for different solvers
% If you have installed mosek, we suggest you use 'Mosek_Conic', which
% solves the 
options.solver = 'Mosek_Conic';
addpath('/Users/Zhe/Downloads/mosek/8/toolbox/r2014a')

% path for simple MKL
% addpath('G:\project\toolbox\simplemkl');

algo = 'eval_MKL_L12'

dataName='diabetes'
mklRes{1}  = feval(algo,dataName, options); % 


