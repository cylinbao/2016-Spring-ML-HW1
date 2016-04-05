#! OCTAVE-INTERPRETER-NAME -qf
clear;

data_size = 50000;

train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
data = train_data.X_train(1:data_size, :);
T_train = train_data.T_train(1:data_size, :);
x1_bound = train_data.x1_bound;
x2_bound = train_data.x2_bound;

sum_vec = [1;1];
% calculate the design matrix
design_mat(:, 1) = data(:, 1:2) * sum_vec;
design_mat(:, 2) = data(:, 1:2).^2 * sum_vec;

% calculate the w for each models and the w0
w_ml = inv(design_mat' * design_mat) * (design_mat') * (T_train);
w0_ml = mean(T_train) - (mean(design_mat) * w_ml);

% save matrix into file train_result
save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat" ...
mean_x1 mean_x2 var_x1 var_x2 design_mat w_ml w0_ml;
