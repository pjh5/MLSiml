function [KMat NWeight] = ConstructMultiKernels(kerneloption, X, X2, Weight); 

% X:     d*n
% KMat:  n*n*k
% NWeight: normalized weight

[dim sz] = size(X);
len_type = size(kerneloption.type, 2); % = {'gaussian' 'gaussian' 'poly' 'poly' };

iKer = 1;

if (~exist('X2','var')) 
    for ti=1:len_type;
        varbs = kerneloption.variablevec{ti};
        types = kerneloption.type{ti};
        paras = kerneloption.optionvect{ti};

        switch (varbs)
            case 'all'
                switch (types)
                    case 'gaussian'
                        options.Kernel = 'rbf';
                        options.libsvm = 1;
                    case 'poly'
                        options.Kernel = 'poly';
                end % types

                for parai=1:size(paras, 2);
                        options.KernelParam = paras(parai);
                        Mat = calkernel(options, X');
                        sD = sum(diag(Mat));
                        KMat(:,:,iKer) = Mat/sD;
                        NWeight(iKer, 1) = sD;
                        iKer = iKer+1;
                end % parai
            case 'single'
                switch (types)
                    case 'gaussian'
                        options.Kernel = 'rbf';
                        options.libsvm = 1;
                    case 'poly'
                        options.Kernel = 'poly';
                end
                for di=1:dim;
                    Xi = X(di, :);

                    for parai=1:size(paras, 2)
                        options.KernelParam = paras(parai);
                        Mat = calkernel(options, Xi');
                        sD = sum(diag(Mat));
                        KMat(:,:,iKer) = Mat/sD;
                        NWeight(iKer, 1) = sD;
                        iKer = iKer+1;
                    end % parai
                end % di
        end % varb
    end % ti
elseif exist('Weight','var') % exist('X2','var') 
    for ti=1:len_type;
        varbs = kerneloption.variablevec{ti};
        types = kerneloption.type{ti};
        paras = kerneloption.optionvect{ti};

        switch (varbs)
            case 'all'
                switch (types)
                    case 'gaussian'
                        options.Kernel = 'rbf';
                        options.libsvm = 1;
                    case 'poly'
                        options.Kernel = 'poly';
                end % types

                for parai=1:size(paras, 2);
                        options.KernelParam = paras(parai);
                        sD = Weight(iKer, 1);
                        Mat = calkernel(options, X', X2');
                        KMat(:,:,iKer) = Mat/sD;
                        iKer = iKer+1;
                end % parai
            case 'single'
                switch (types)
                    case 'gaussian'
                        options.Kernel = 'rbf';
                        options.libsvm = 1;
                    case 'poly'
                        options.Kernel = 'poly';
                end
                for di=1:dim;
                    Xi = X(di, :);
                    Xi2 = X2(di, :);

                    for parai=1:size(paras, 2)
                        options.KernelParam = paras(parai);
                        sD = Weight(iKer, 1);
                        Mat = calkernel(options, Xi', Xi2');
                        KMat(:,:,iKer) = Mat/sD;
                        iKer = iKer+1;
                    end % parai
                end % di
        end % varb
    end % ti
    NWeight = [];
else
    disp('Needs four input parameters');
end

