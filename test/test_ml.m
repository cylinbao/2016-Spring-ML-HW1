#! OCTAVE-INTERPRETER-NAME -qf
clear;

data_size = 10000;
test_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Test_data1_hw1.mat');
train_result = load('~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat');

data = test_data.X_test(1:data_size, :);
t_data = test_data.T_test(1:data_size);
mean_x1 = train_result.mean_x1;
mean_x2 = train_result.mean_x2;
var_x1 = train_result.var_x1;
var_x2 = train_result.var_x2;
w_ml = train_result.w_ml;
w0 = train_result.w0;

sum_mse_ml = (~data(:,3))' * (exp(-((data(:, 1) - mean_x1).^2./(2*var_x1)) ...
						-((data(:, 2) - mean_x2).^2./(2*var_x2)))*w_ml + w0 - t_data).^2;

land_num = sum(data(:,3)==0);
mse_ml = sqrt(sum_mse_ml/land_num);

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/test/test_result.mat" mse_ml;
