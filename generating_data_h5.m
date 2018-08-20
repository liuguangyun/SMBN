%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is used to generate the training and test samples which 
% are saved as hdf5 format for the caffe input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc;
addpath('drtoolbox');
addpath('drtoolbox/gui');
addpath('drtoolbox/techniques');

savepath_train = './data_h5/indian/train.h5';
savepath_test  = './data_h5/indian/test.h5';


%%%%%%%%%%%%%%%%%  Load dataset and groundtruth %%%%%%%%%%%%%%%%%%
%% for Indian Pines, the size of input is 23*23*5 %%%%%%%%%%%%%%%%
load datasets/Indian_pines_corrected.mat;
img=indian_pines_corrected;
load datasets/Indian_pines_gt.mat;

pad_size=11;
num_bands=5;
input_size=2*pad_size+1;
no_classes=16;

per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];      % 10% sampling. test: 9220  train: 1029
[I_row,I_line,I_high] = size(img);
img=reshape(img,I_row*I_line,I_high);

%%%%%%%%%%%%%%  Dimension-reducing by PCA  %%%%%%%%%%%%%%%%%%%%%%%
im=img;
im=compute_mapping(im,'PCA',num_bands);
im = mat2gray(im);
im =reshape(im,[I_row,I_line,num_bands]);

%%%%%  scale the image from -1 to 1
im=reshape(im,I_row*I_line,num_bands);
[im ] = scale_func(im);
im =reshape(im,[I_row,I_line,num_bands]);
%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size,pad_size],'symmetric');

%%%%%% choose the training and test samples randomly  %%%%%%%%
Train_Label = [];
Train_index = [];
train_data=[];
test_data=[];
train_label=[];
test_label=[];
train_index=[];
test_index=[];
index_len=[];

for ii = 1: no_classes
    
   index_ii =  find(indian_pines_gt == ii)';
   index_len=[index_len length(index_ii)];
   labeled_len=ceil(index_len*0.07);
   rand_order=randperm(length(index_ii));
   class_ii = ones(1,length(index_ii))* ii;
   Train_Label = [Train_Label class_ii];
   Train_index = [Train_index index_ii]; 
   
   num_train=per_class_num(ii);
   train_ii=rand_order(:,1:num_train);
   train_index=[train_index index_ii(train_ii)];

   test_index_temp=index_ii;
   test_index_temp(:,train_ii)=[];
   test_index=[test_index test_index_temp];

   train_label=[train_label class_ii(:,1:num_train)];
   test_label=[test_label class_ii(num_train+1:end)];
   
end

%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(input_size,input_size,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size;
            Y_new = Y+pad_size;
            X_range = [X_new-pad_size : X_new+pad_size];
      
            Y_range = [Y_new-pad_size : Y_new+pad_size]; 

            img_data=im_extend(X_range,Y_range,:);
            
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
order = randperm(count);
TRAIN_INDEX=train_index(:,order);
TRAIN_DATA=TRAIN_DATA(:,:,:,order);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(input_size,input_size,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size;
            Y_new = Y+pad_size;
            X_range = [X_new-pad_size : X_new+pad_size];
      
            Y_range = [Y_new-pad_size : Y_new+pad_size]; 

            img_data=im_extend(X_range,Y_range,:);
            
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
order = randperm(count);
TEST_INDEX=test_index(:,order);
TEST_DATA=TEST_DATA(:,:,:,order);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order);
data=TEST_DATA;
label=TEST_LABEL;
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

