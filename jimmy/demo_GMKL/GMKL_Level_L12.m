function [theta,w,bsvm,posw,fval,history] = GMKL_Level_L12(K,yapp,C,option,verbose);
%  Haiqin Yang
%  All copyrights reserved.

% number of kernels
nbkernel=size(K,3);
% number of training data
n=size(yapp,1);
% maximum number of iterations

if ~isfield(option,'nbitermax');
    nbitermax=1000;
else
    nbitermax=option.nbitermax;
end;
% the criterion: difference of sigmas (kernel weights)
if ~isfield(option,'seuildifftheta');
    option.seuildifftheta=1e-4;
end
% 
if ~isfield(option,'seuildiffconstraint');
    option.seuildiffconstraint=0.05;
end
% the criterion: duality gap
if ~isfield(option,'seuildualitygap');
    option.seuildualitygap=0.01;
end
%
if ~isfield(option,'lambdareg');
    lambdareg=1e-10;
    option.lambdareg=1e-10;
else
    lambdareg=option.lambdareg;
end

if ~isfield(option,'exitcon');
    option.lambdareg=1e-6;
end

% whether show the output of algorithms during iterations
if ~isfield(option,'verbosesvm');
    verbosesvm=0;
    option.verbosesvm=0;
else
    verbosesvm=option.verbosesvm;
end

% whether initialize sigma
if ~isfield(option,'thetainit');
    rv = roots([1-option.v, option.v, -1/nbkernel]);
%     c0 = option.v*nbkernel+(1-option.v)*nbkernel^0.5;
    if (rv(1)>0)
        c0 = rv(1);
    else
        c0 = rv(2);
    end
    theta=ones(nbkernel,1)*c0;
else
    theta=option.thetainit ;
    if size(theta,1)==1
        theta=theta';
    end;
end;
%
if ~isfield(option,'seuil');
    seuil=1e-12;
else
    seuil=option.seuil;
end
% whether initialize alpha
if isfield(option,'alphainit');
    alphainit=option.alphainit;
else
    alphainit=[];
end;
% the parameter of level MKL
if isfield(option,'lambda');
    AlgLambda=option.lambda;
else
    AlgLambda= 0.99;
end;

kernel='numerical';
span=1;

teldefvalold = inf;
fvalold = -inf;
%---------------------------------------------------------------------
%% Setting the linear prog  parameters
% nbvar = nbkernel+1;
%
% var = [theta theta1, theta_2,   ..., thetaK];
%---------------------------------------------------------------------
%% 
nu = -inf;
f=[1;zeros(nbkernel+1,1)];
% Aeq=[0 ones(1,nbkernel)]; 
% beq= sumtheta;
A=[];
b=[];

A2 = [];
b2 = [];
%% QP
QH = eye(nbkernel);
Qf = zeros(nbkernel,1);
QAeq = ones(1,nbkernel);
% Qbeq = sumtheta;
QLB=[zeros(nbkernel,1)];
QUB=[ones(nbkernel,1)];
QA = [];
Qb = [];
Qb2 = [];

% How to change the parameters that controls
% the accuracy of a solution computed by the conic
% optimizer
param = [];

% Primal feasibility tolerance for the primal solution
param.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1.0e-8;

% Dual feasibility tolerance for the dual solution
param.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1.0e-8;

% Relative primal-dual gap tolerance.
param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1.0e-8;

%% Common
%optimopt=optimset('MaxIter',500,'Display','off', 'TolCon',1e-3,'TolFun',1e-5);
thetaold=theta;
teldetheta = theta;
teldexold = [-inf; theta];
x=[];
exitflag=0;

% theta is the weight vector
history.obj_t=[];
history.theta=[];
history.KKTconstraint=[1];
history.cpgap=[];
history.theta= [history.theta;theta'];

kerneloption.matrix=sumKbeta(K,theta);

% grad is the gradient
history.grad = [];
history.lowbound = [];
history.upbound = [];


nbverbose=1;
iter=0;
loop=1;

history.timecost = [];
t0= cputime;

while loop
    
    %----------------------------------
    %%0 run SVM
    % You may use other types of solves
    %----------------------------------
    kerneloption.matrix=sumKbeta(K,theta);
    if  iter >0; %~isempty(alphainit) &
        alphainit=zeros(size(yapp));
        alphainit(posw)=alphaw;
    end;
    % solver of svm
    [xsup,w,bsvm,posw,timeps,alphaw,obj]=svmMKLclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);
    
    %--------------------------------------------------------
    % gradient
    %--------------------------------------------------------
    % w = alphaw.*y;
    for i=1: nbkernel
        grad(i)= -0.5*w'*K(posw,posw,i)*w;% - sum(alphaw);
    end

    obj_t = double(obj);
    history.grad=[history.grad; grad];
    history.obj_t=[history.obj_t obj_t];
    
    grad = double(grad); %
    %--------------------------------------------------------
    % duality gap
    %--------------------------------------------------------    
    % calculate constrain violation
%     dualobj = -max(-grad)/(2*(1-option.v)*max(theta)+option.v)+sum(alphaw);
%     dualobj = -max(-grad'./(option.v+2*(1-option.v)*theta*(1+(1-option.v)*sum(theta.*theta))))+sum(alphaw);
%     dualitygap=(obj-dualobj)/obj;
% 
%     if (option.v==0)
%         dualitygap = abs(dualitygap);
%     end
    
    % store duality gap
%     history.dualitygap=[history.dualitygap dualitygap];

    %-----------------------------------------------------
    %       Maximum constraint violation check
    %------------------------------------------------------
    % S is the minimum of the objective
%     S= grad*theta+sum(alphaw);
    indpos=find(theta>option.numericalprecision);
    indzero=find(abs(theta<=option.numericalprecision));

    KKTconstraint=abs ((min(grad(indpos))-max(grad(indpos)))/min(grad(indpos))) ;
    KKTconstraintZero=  (min(grad(indzero))>max(grad(indpos)) );
    
%     KKTconstraint=abs(S/obj_t-1);
    history.KKTconstraint=[history.KKTconstraint KKTconstraint];
    
    %----------------------------------------------------
    %  Optimize the weigths theta using a LP
    %----------------------------------------------------
    %% 1. calculate the lower bound
    % II solve tilde_p
    % -S is the gradient
    % obj_t is the obj
    thetaold=theta;
    
    A=[A;-1 grad];
    A2 = [A2; [-1 0 0 grad]];
    b2 = [b2; -sum(alphaw)];
    
    % the cutting plane constraint
    aux = -sum(alphaw); %grad*theta-obj_t;
    b=[b;aux];

    blc   = [-inf*ones(length(b),1)];
    buc   = [b];
    
    switch option.solver
        case 'CVX'
            cvx_begin
                variables x(nbkernel+1)
                minimize x(1)
                subject to 
                    A(:,1:nbkernel+1)*x<=b;
                    option.v*sum(x(2:nbkernel+1))+(1-option.v)*(x(2:nbkernel+1)'*x(2:nbkernel+1))<=1;
                    x(2:nbkernel+1)>=0;
            cvx_end
%             save tmp A b x
            exitflag = strcmp(cvx_status, 'Solved') | strcmp(cvx_status, 'Inaccurate/Solved') ;
        case 'Yalmip'
            xi = sdpvar(1,1);
            x = sdpvar(nbkernel,1);
            objective = xi;
            F = set(A*[xi;x]<=b);
            F = F+set(option.v*ones(1,nbkernel)*x+(1-option.v)*x'*x<=1);
            F = F+set(x>=0);
            ops = sdpsettings('verbose',1,'solver','mosek'); %sedumi
            sol = solvesdp(F, objective, ops);
            xx = double(x);
            x = double([xi; x]);
            exitflag = abs(option.v*sum(xx)+(1-option.v)*xx'*xx-1)<option.exitcon;
        case 'Mosek'
            % using MOSEK
            clear prob
            prob.c = [1;zeros(nbkernel,1)];
            prob.qcsubk = [(size(A,1)+1)*ones(nbkernel+1,1)];
            prob.qcsubi = [1:nbkernel+1]';
            prob.qcsubj = [1:nbkernel+1]';
            prob.qcval = [0 2.0*(1-option.v)*ones(1, nbkernel)]';
            
            prob.a = sparse([A; [0 ones(1, nbkernel)*option.v]]);

            % Specify lower bounds of the constraints.
%             prob.blc = [-inf(size(b,1)); 1];
            
            % Specify  upper bounds of the constraints.
            prob.buc = [b; 1];
            
            % Specify lower bounds of the variables.
            prob.blx = [-inf; zeros(nbkernel,1)];
            
            % Specify upper bounds of the variables.
            prob.bux = [inf*ones(nbkernel+1,1)];            

            % Perform the optimization.
            [r,res]       = mosekopt('minimize echo(0)',prob,param);
            x=res.sol.itr.xx;              
            xx = x(2:end);
            exitflag = strcmp(res.sol.itr.solsta, 'OPTIMAL') | abs(option.v*sum(xx)+(1-option.v)*xx'*xx-1)<option.exitcon | x(1)>nu;
%             ;
            
%             if (exitflag==0) % 
%                 xx = x(2:end,1);
%                 xx(find(xx<option.numericalprecision)) = 0;      
%                 
                % numerical problem
%                 quad_con = option.v*sum(xx)+(1-option.v)*xx'*xx; 
%                 while (quad_con>1) % if the constraint of L1+L2<=1 violates
%                     II = find(xx>0);
%                     [gsort, Isort] = sort(-grad(II), 'ascend'); 
%                     
%                     x_sel = xx(II(Isort(1)));
%                     rv = roots([1-option.v, -(option.v+2*(1-option.v)*x_sel), quad_con-1]);
%                     val = min(rv);
%                     
%                     if (val<x_sel)
%                         xx(II(Isort(1))) = xx(II(Isort(1)))-val;
%                     else
%                         xx(II(Isort(1))) = 0;
%                     end
%                     quad_con = option.v*sum(xx)+(1-option.v)*xx'*xx; 
%                 end
%                 bAx = -b+A(:,2:end)*xx;
%                 x = [max(bAx); xx];
%             end
        %   disp(strcat('Lowbound Function:', res.sol.itr.solsta));
        case 'Mosek_Conic'
            clear prob
            prob.c = [1;zeros(nbkernel+2,1)];
            prob.a = sparse([A2; [0, (1-option.v), 0, ones(1, nbkernel)*option.v]]);

            % Specify  lower bounds of the constraints.    
            prob.blc = [-inf*ones(size(A2,1),1); 1];

            % Specify  upper bounds of the constraints.    
            prob.buc = [b2; 1];

            % Specify lower bounds of the variables.                
            prob.blx = [-inf; 0; 0.5; zeros(nbkernel,1)];

            % Specify upper bounds of the variables.
            prob.bux = [inf; inf; 0.5; inf*ones(nbkernel,1)];

            % Specify the cones;
            prob.cones   = cell(1,1);

            % The first cone is specified.
            prob.cones{1}.type = 'MSK_CT_RQUAD';
            prob.cones{1}.sub  = [2, 3, 4:nbkernel+3];
            
            % Optimize the problem. 
            [r,res]=mosekopt('minimize echo(0)',prob);

            % Display the primal solution.
            x = res.sol.itr.xx;
            exitflag = strcmp(res.sol.itr.solsta, 'OPTIMAL');
    end
    
    if (exitflag)%strcmp(res.sol.itr.solsta, 'OPTIMAL') %~isempty(x)
%         if (x(1)>teldexold(1))
            % the lower bound
        nu = x(1);
        teldetheta= x(2:end);
        teldexold= x;
        teldefvalold = nu;
    else
        nu = teldexold(1);
        teldetheta= teldexold(2:end);
        teldefval = teldefvalold;
        
%         if (option.v~=1 || option.v~=0)
            loop=0;
%         end;
        fprintf(1,'Premature convergence of the algorithm \n');
    end;
    
     %% 2. calculate the bound
    lowbound = min(nu,obj_t);
    upbound = min(history.obj_t);
    cpgap = upbound-lowbound;
    history.cpgap = [history.cpgap; cpgap];
    history.lowbound = [history.lowbound; lowbound];
    history.upbound = [history.upbound; upbound];
    
    if abs(cpgap/upbound) < 1e-2
        AlgLambda = option.lambdaFinal;
    end
    
    %% 3. project to the solid
    %----------------------------------------------------
    %  project to the solid using QP
    %----------------------------------------------------
    if nu < obj_t
        Qf = -theta;    
        QA =[QA; grad];
        % the cutting plane constraint
%         aux = (1- AlgLambda)*lowbound+ AlgLambda * upbound - obj_t + grad*theta;
        Lt = (1- AlgLambda)*lowbound+ AlgLambda*upbound;
        Qb = [Qb; Lt-sum(alphaw)];
        Qb2 = [Qb2; Lt-sum(alphaw)-grad*theta];
%         blc   = [-inf*ones(length(Qb),1)];
%         buc   = [Qb];
        
        ss = option.solver;
%         option.solver = 'CVX';
        switch option.solver
            case 'CVX'
                cvx_begin
                    variables x(nbkernel)
                    minimize norm(x-theta)
                    subject to 
                        option.v*sum(x)+(1-option.v)*x'*x<=1;
                        QA*x<=Qb;
                        x>=0;                        
                cvx_end
                cvx_x = x;
%                 save(['CVX_', num2str(iter), '.mat'], 'QA', 'Qb', 'cvx_x', 'theta');
                
                exitflag = strcmp(cvx_status, 'Solved');
                option.solver = ss;                
            case 'Yalmip'
                tic;
                x = sdpvar(nbkernel,1);
                objective = norm(x-theta);
                F = set(QA*[x]<=Qb);
                F = F+set(option.v*ones(1,nbkernel)*x+(1-option.v)*x'*x<=1);
                F = F+set(x>=0);
                ops = sdpsettings('verbose',1,'solver','mosek'); %sedumi
                sol = solvesdp(F, objective, ops);
                xx = double(x);
                toc
                exitflag = abs(option.v*sum(xx)+(1-option.v)*xx'*xx-1)<option.exitcon;
            case 'Mosek'
                clear prob
%                 tic;
                prob.c = [1;zeros(nbkernel,1)];
                prob.qcsubk = [ones(nbkernel+1,1)', 2*ones(nbkernel+1,1)']';
                prob.qcsubi = [1:nbkernel+1, 1:nbkernel+1]';
                prob.qcsubj = [1:nbkernel+1, 1:nbkernel+1]';
                prob.qcval = [[0 2*(1-option.v)*ones(1, nbkernel)]', [0 2*ones(1, nbkernel)]'];

                prob.a = sparse([[0 ones(1, nbkernel)*option.v]; [-1 -2*theta']; [zeros(size(QA, 1),1) QA]]);
    
                % Specify  upper bounds of the constraints.    
%                 prob.blc = [1; 0; -inf*size(Qb,1)];

                % Specify  upper bounds of the constraints.    
                prob.buc = [1; -theta'*theta; Qb];

                % Specify lower bounds of the variables.                
                prob.blx = [0; zeros(nbkernel,1)];

                % Specify upper bounds of the variables.
                prob.bux = [inf*ones(nbkernel+1,1)];

                [r,res]       = mosekopt('minimize echo(0)',prob);       
                xx = res.sol.itr.xx;
                x = xx(2:end);
                exitflag = strcmp(res.sol.itr.solsta, 'OPTIMAL') | abs(option.v*sum(x)+(1-option.v)*x'*x-1)<option.exitcon;
                
%                 toc
%                 save(['Mosek_', num2str(iter), '.mat'], 'QA', 'Qb', 'x');
                
            case 'Mosek_Conic'  % results the same as cvx
                clear prob
%                 tic;
                prob.c = [1;zeros(nbkernel+2,1)];
                prob.a = sparse([[zeros(size(QA, 1),3), QA]; ...
                                 [-1, 1, 0, -2*theta']; ...
                                 [0, (1-option.v), 0, option.v*ones(1,nbkernel)]]);
    
                % Specify  lower bounds of the constraints.    
                prob.blc = [-inf*ones(size(Qb,1),1); -inf; 1];

                % Specify  upper bounds of the constraints.    
                prob.buc = [Qb; -theta'*theta; 1];

                % Specify lower bounds of the variables.                
                prob.blx = [0; 0; 0.5; zeros(nbkernel,1)];

                % Specify upper bounds of the variables.
                prob.bux = [inf; inf; 0.5; inf*ones(nbkernel,1)];

                % Specify the cones;
                prob.cones   = cell(1,1);

                % The second cone is specified.
                prob.cones{1}.type = 'MSK_CT_RQUAD';
                prob.cones{1}.sub  = [2:nbkernel+3];
                
%                 tic;
                [r,res]       = mosekopt('minimize echo(0)',prob);       
%                 toc
                exitflag = strcmp(res.sol.itr.solsta, 'OPTIMAL');
                xx = res.sol.itr.xx;
                x = xx(4:end,1);
%                 toc
                %}
%                 save(['Mosek_', num2str(iter), '.mat'], 'QA', 'Qb', 'x');
                
        end
        fval = x'*x;
        if (exitflag) %~isempty(x)            
            theta= x;
            theta(find(theta<option.numericalprecision))=0;            
            xold= x;
            fvalold=fval;
        else
            theta= xold;
            fval=fvalold;
            loop=0;
            fprintf(1,'Premature convergence of the algorithm \n');
        end;
    end
    
    %-------------------------------
    % Numerical bounding and cleaning
    %-------------------------------    
    if seuil ~=0 & max(theta)>seuil & nloop < option.seuilitermax
        theta=(theta.*(theta>seuil))*Sumtheta/sum(theta.*(theta>seuil));
    end;

%     numKer = length(find(theta>option.numericalprecision));

    history.theta= [history.theta;theta'];
%     history.lowbound = [history.lowbound lowbound];
%     history.upbound = [history.upbound upbound];
    


    %------------------------------------------------------
    % verbosity
    %----------------------------------------------------
   

    if verbose ~= 0
        if nbverbose == 1
            disp('------------------------------------------------');
            disp('iter    obj    Deltatheta  lowerbound   upperbound  cpgap KKT');
            disp('------------------------------------------------');
        end
        if nbverbose == 20
            nbverbose=1;
        end

        if exitflag==0
            fprintf('%d   | %8.4f | %6.4f | %6.4f | %6.4f | %6.4f | %6.4f \n',[iter obj_t   max(abs(theta-thetaold)) lowbound upbound cpgap KKTconstraint]);
        else
            fprintf('%d   | %8.4f | %6.4f | %6.4f | %6.4f | %6.4f | %6.4f \n',[iter obj_t   max(abs(theta-thetaold)) lowbound upbound cpgap KKTconstraint]);
        end;
        nbverbose = nbverbose+1;
    end
    
   
    %----------------------------------------------------
    % check variation of theta conditions
    %----------------------------------------------------
    if  option.stopvariation==1 & option.stopKKT==0 & max(abs(theta - thetaold))<option.seuildifftheta;
        loop=0;
        fprintf(1,'variation convergence criteria reached \n');
    end;

    %----------------------------------------------------
    % check KKT conditions
    %----------------------------------------------------
    if   option.stopKKT==1 &  option.stopvariation==0 & ( KKTconstraint < option.seuildiffconstraint )
        loop = 0;
        fprintf(1,'KKT (maximum violation constraint) convergence criteria reached \n');
    end;

    %----------------------------------------------------
    % check KKT and variation of theta conditions
    %----------------------------------------------------
    if  option.stopvariation== 1 & option.stopKKT==1 & max(abs(theta - thetaold))<option.seuildifftheta &  ( KKTconstraint < option.seuildiffconstraint )
        loop = 0;
        fprintf(1,'variation and KKT convergence criteria reached \n');
    end;
    %----------------------------------------------------
    % check for duality gap
    %----------------------------------------------------
%     if option.stopdualitygap==1 && dualitygap<option.seuildualitygap
%         loop=0;
%         fprintf(1,'Duality gap of primal-dual \n');        
%     end;


    %----------------------------------------------------
    % check nbiteration conditions
    %----------------------------------------------------
    if iter>=option.nbitermax 
        loop = 0;
        fprintf(1,'Maximal number of iterations reached \n');
    end;

    %----------------------------------------------------
    % check for cutting plane gap
    %----------------------------------------------------
    if option.stopapproxgap && cpgap < option.seuilboundgap
        loop=0;
        fprintf(1,'The gap between the lowerbound and the upperbound reached \n');
    end;
    
    %--------------------------------------------
    % time
    %--------------------------------------------
    iter=iter+1;
end;
history.timecost = [cputime-t0];



