#! OCTAVE-INTERPRETER-NAME -qf
clear;

data_size = 50000;

train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
data = train_data.X_train(1:data_size, :);
T_train = train_data.T_train(1:data_size, :);
x1_bound = train_data.x1_bound;
x2_bound = train_data.x2_bound;

design_mat(:, 1:2) = data(:, 1:2);
design_mat(:, 3:4) = data(:, 1:2).^2;

% using ml approach 
w_ml = inv(design_mat' * design_mat) * (design_mat') * (T_train);
w0_ml = mean(T_train) - (mean(design_mat) * w_ml);

% using map appraoch 
lambda = -91.3;
reg_mtx = lambda * diag(ones(1,4));
w_map = inv( reg_mtx + design_mat' * design_mat) * (design_mat') * (T_train);
w0_map = mean(T_train) - (mean(design_mat) * w_map);

% using bayesian approach
pkg load statistics
alpha_num = 0.1;
beta_num = 0.1;

S_mtx = inv(alpha_num*diag(ones(1,4)) + beta_num*design_mat'*design_mat);
M_mtx = beta_num * S_mtx * design_mat' * T_train;

w_baye = (mvnrnd(M_mtx, S_mtx))';
w0_baye = mean(T_train) - (mean(design_mat) * w_baye);
pkg unload statistics

% save matrix into file train_result
save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat" ...
w_ml w0_ml w_map w0_map w_baye w0_baye;
