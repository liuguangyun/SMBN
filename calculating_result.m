%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is used to calculate three metrics (i.e., OA, AA, Kappa) 
% and draw classification map.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc

OA=[];
AA=[];
CA=[];
kappa=[];

%%%%%%%%%%%%%%%% for the Indian Pines image  %%%%%%%%%%%%%%%%%%%%%%
num_classes=16;   
prob_data=importdata('info/indian_pines_prob.txt');
load TRAIN_INDEX.mat;
load TRAIN_LABEL.mat;
load TEST_INDEX.mat;
load TEST_LABEL.mat;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prob_data_temp=prob_data;  
true_label=TEST_LABEL';
test_prob=[];
predict_label=[];
for i=1:length(prob_data_temp)/num_classes
    prob_temp=prob_data_temp((i-1)*num_classes+1:i*num_classes,:)';
    test_prob=[test_prob;prob_temp];
    predict_label=[predict_label;find(prob_temp==max(prob_temp))];
end

[OA,kappa, AA,CA]=calcError(true_label,predict_label-1,1:num_classes)

%%%%%%%%%%%%%%%%%%%%%%  draw classification map    %%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%  for the Indian Pines image %%%%%%%%%%%%%%%%
resultsmap=zeros(145*145,1);  
resultsmap(TRAIN_INDEX',:)=TRAIN_LABEL'+1;
resultsmap(TEST_INDEX',:)=predict_label;
resultsmap=reshape(resultsmap,145,145);
maps=label2color(resultsmap,'india');  
imwrite(maps,'SMBN_Indian.bmp','bmp'); 
figure,imshow(maps);

