% dim 3 data
knn3.name = 'knn3';
knn3.value = 	[88.2	83.1	79.4	78.4	76.1	76.8	75	70.9	67.3	66.8	66.9	67.4	61.8	58.8	58.5	57.9	57.1	55.3	52.5	52.2	48.8];

rf3.name = 'random forest 3';
rf3.value = 	[84	73.8	75	72	74.5	69.9	70.5	66.6	59.8	62.7	60.2	58.2	58	55.7	57	57.4	52.9	49.1	51	52.3	48];

svm3.name = 'svm3';
svm3.value = 	[85.3	84.5	79.6	77.2	78	77.5	75.2	72.3	67	67.2	68.8	68.9	63.3	59.3	60.9	60	59	54.3	54.8	52.8	50.7];

mkl3.name = 'mkl';
mkl3.value = [99.7	98.62	98.56	98.14	98.32	97.84	89.74	89.14	89.082	89.14	89.61	79	78.4	79.24	78.04	70.25	68.99	68.25	58.52	57.2	49.88];

mklcon3.name = 'concatenated mkl';
mklcon3.value = [98.5	93.7	90.64	89.26	89.68	92.44	83.38	82	76.36	77.01	75.87	73	70.19	65.81	62.39	63.17	59.06	55	56.54	54.26	48.29];

mklsingle3.name = 'mkl single stream';
mklsingle3.value = 	[98.08	98.08	98.08	98.08	98.08	98.08	88.06	88.06	88.06	88.06	88.06	77.86	77.86	77.86	77.86	69.15	69.15	69.15	55.31	55.31	48.1104];



% dim 5 data


knn5.name = 'knn5';
knn5.value = [61.5	61.2	57.6	55.2	57.3	57.3	58.2	54.7	54.3	54.8	54.9	55.3	54.2	50.9	50.4	52.4	52.2	53.3	51	50.7	49.1];
rf5.name = 'Random Forest 5';
rf5.value = [58.6	56.8	58.3	54.9	56.7	57.4	55.7	53.2	53.9	53.5	54.3	50.8	51.6	51.5	50.7	48.1	51.7	50	52.4	48.4	49.7];
svm5.name = 'SVM 5';
svm5.value = [62.6	62	59.5	57	59.7	57.4	59.5	57.3	55.1	54.7	53.8	55.4	55.6	49.7	52.5	50.9	51.2	48.6	53.1	50.8	49];
mkl5.name = 'mkl 5';
mkl5.value  = [94.72	92.32	88.84	88.6	91.24	91.06	85.53	80.74	79.18	81.57	79.72	68.27	71.09	63.11	59.57	63.59	62.42	56.63	53.66	52.19	48.47];
mklsingle5.name = 'mkl single stream';
mklsingle5.value = [92.26	92.26	92.26	92.26	92.26	92.26	78.7	78.7	78.7	78.7	78.7	65.8	65.8	65.8	65.8	57.35	57.35	57.35	55.24	55.24	49.97];
mklcon5.name = 'mkl concatenation 5';
mklcon5.value = [67.67	65.75	60.38	58.61	60.83	55.97	62.45	61.25	56.87	54.95	60.44	60.95	55.43	56.51	54.89	54.26	53.99	52.28	49.97	48.35	48.83];

% dim 7 data

knn7.name = 'knn7';

knn7.value =	[54.59	54.29	55.01	54.35	52.67	56.09	53.92	54.77	50.57	53.57	51.95	52.37	50.99	50.51	52.79	51.95	51.05	48.59	51.29	50.45	51.83];
rf7.name = 'Random Forest 7';
rf7.value = [54.29	51.17	52.85	53.75	49.31	52.55	52.25	51.53	52.31	50.81	50.81	53.33	51.17	48.65	48.53	50.39	52.73	50.75	49.19	47.81	49.79];
svm7.name = 'SVM 7';
svm7.value = [	57.05	54.35	55.91	53.21	50.87	54.05	56.39	54.71	52.61	55.37	49.31	53.93	50.15	52.07	50.99	50.63	51.95	50.15	51.89	48.05	49.67];
mkl7.name = 'mkl 7 ';
mkl7.value = [72.57	68.19	65.6	65.97	63.87	69.64	65.45	61.37	58.55	58.55	62.27	61.36	59.18	56.78	57.71	56.18	54.05	54.23	48.92	50.36	50.24];
mklsingle7.name = 'mkl single stream 7';

mklsingle7.value = [64.55	64.55	64.55	64.55	64.55	64.55	60.02	60.02	60.02	60.02	60.02	58.73	58.73	58.73	58.73	53.57	53.57	53.57	52.85	52.85	46.07];
mklcon7.name = 'mkl concatenation 7';
mklcon7.value =	[63.99	60.11	59.12	55.97	54.89	57.47	58.73	58.85	56.6	55.1	55.43	56	53.99	56.75	54.77	52.94	51.8	53.51	49.58	50	52.13];


%toplot = [mkl3, mklcon3, mklsingle3, rf3, knn3, svm3];
 toplot = [mkl5, mklcon5, mklsingle5];
% toplot = [mkl7, mklcon7, mklsingle7, rf7, knn7, svm7];
N = length(toplot);
col_num = ceil(N /2);
% subplot(2, col_num);
collim = [0, 100];

for i = 1: N

    subplot(1,3, i )
    cur_struct = toplot(i);
    cur_name = cur_struct.name
    cur_vec = cur_struct.value;
    num_rows = floor(sqrt(length(cur_vec)* 2));
    cur_mat = triu(ones(num_rows));
    cur_mat = cur_mat';
    myfloor = collim(1);
    cur_mat(cur_mat==1) = cur_vec
    cur_mat(cur_mat == 0) = myfloor + 10 ;
    cur_mat = cur_mat;
    
    surf(cur_mat)
%     zlim([0, 150])
%     hold on 
    im = imagesc(cur_mat, collim)
%     im.XData = [0, 5]
%     im.YData = [0, 5]
%     hold off
%     im  = imagesc(cur_mat, collim);
%     im.AlphaData = .7;

    colorbar()
    title(cur_name, 'FontSize', 20)
    %axis([0.5 5.5 0.5 5.5])
    set(gca,'fontsize',15)
end