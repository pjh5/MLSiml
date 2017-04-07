function [Weigth,InfoKernel]=UnitTraceNormalization(x,kernelvec,kerneloptionvec,variablevec)
% describe the kernel type, kernel parameter, comprising variables, kernel weight for each single kernel
% INPUT: 
%   x: duh
%   kernelvec: all kernel type
%   kerneloptionvec: all kernel params 
%   variablevec: variable used for kernel generation

% OUTPUT:
%   Weight: a list of wighting to obtain unit trace for each kernel
%   InfoKernel: NumKernels x all information required to generate the
%               single kernel
chunksize=200;
N=size(x,1);
nbk=1;
for i=1:length(kernelvec);
    % i
    for k=1:length(kerneloptionvec{i})

        somme=0;

        chunks1=ceil(N/chunksize);

        for ch1=1:chunks1
            ind1=(1+(ch1-1)*chunksize) : min( N, ch1*chunksize);
            somme=somme+sum(diag(svmkernel(x(ind1,variablevec{i}),kernelvec{i},kerneloptionvec{i}(k))));
        end;
        %         for j=1:N
        %             somme=somme+svmkernel(x(j,variablevec{i}),kernelvec{i},kerneloptionvec{i}(k));
        %
        %         end
        if somme~=0
            Weigth(nbk)=1/somme;
            InfoKernel(nbk).kernel=kernelvec{i};
            InfoKernel(nbk).kerneloption=kerneloptionvec{i}(k);
            InfoKernel(nbk).variable=variablevec{i};
            InfoKernel(nbk).Weigth=1/somme;
            nbk=nbk+1;
%         else
%             A
        end;
    end;
end;