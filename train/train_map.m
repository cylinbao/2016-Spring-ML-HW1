#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [14, 18];
block_size = 50;
data_size = 50000;
lambda = 0.7;

train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
data = train_data.X_train(1:data_size, :);
T_train = train_data.T_train(1:data_size, :);

train_result = load('~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat');
mean_x1 = train_result.mean_x1;
mean_x2 = train_result.mean_x2;
var_x1 = train_result.var_x1;
var_x2 = train_result.var_x2;
design_mat = train_result.design_mat;

% calculate the w for each models and the w0
diagmtx = diag(ones(1, block_num(1)*block_num(2)));
w_map = inv(lambda*diagmtx + design_mat'*design_mat) * (design_mat') * (T_train);
w0_map = mean(T_train) - (mean(design_mat) * w_map);

% save matrix into file train_result
save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat" ...
w_map w0_map;
