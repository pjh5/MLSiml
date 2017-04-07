function [kernelcellaux,kerneloptioncellaux,variablecellaux]=CreateKernelListWithVariable(variablecell,dim,kernelcell,kerneloptioncell)
% INPUT:
% variablecell: cell of all kernel generation method,
% all, use all features
% single, gnerate a kernel for each individual vector
% random, generate a  kerne for a random subset of the feature set
% kernel cell: cell of th eaxact type of kernel
% kernel option: list of parameters usde to generate a kernal
% kernel_indices:

% OUTPUT:
% a list with # kernel generation techniques to be used kernelcellaux
%                                                       Cartisian X
%                                                       kerneloptionaux
% kernelcellaux: kernel type
% kerneloptioncellaux: kernel param
% variablecellaux: variable used to generate current kernel
j=1;

% for each meta kerel generation class
for i=1:length(variablecell)
%     class(variablecell{i})
%     isa(variablecell{i}, 'int64')
    if isa(variablecell{i}, 'int64')
        disp('hellooooooo')
        kernelcellaux{j}=kernelcell{i};
        kerneloptioncellaux{j}=kerneloptioncell{i};
        indicerand=randperm(dim);
        nbvarrand=floor(rand*dim)+1;
        variablecellaux{j}=variablecell{i};
        j=j+1;
    else
        switch variablecell{i}
            
            case 'all'
                kernelcellaux{j}=kernelcell{i};
                kerneloptioncellaux{j}=kerneloptioncell{i};
                variablecellaux{j}=1:dim;
                j=j+1;
                
            case 'single'
                for k=1:dim
                    kernelcellaux{j}=kernelcell{i};
                    kerneloptioncellaux{j}=kerneloptioncell{i};
                    variablecellaux{j}=k;
                    j=j+1;
                end;
            case 'random'
                kernelcellaux{j}=kernelcell{i};
                kerneloptioncellaux{j}=kerneloptioncell{i};
                indicerand=randperm(dim);
                nbvarrand=floor(rand*dim)+1;
                variablecellaux{j}=indicerand(1:nbvarrand);
                j=j+1;                                
            otherwise

                warning('unrecognized kernel indice information')
        end
        
    end
    
end
