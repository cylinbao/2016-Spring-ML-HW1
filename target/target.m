#! OCTAVE-INTERPRETER-NAME -qf
clear;

data_size = 10000;
test_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Test_data1_hw1.mat');
train_result = load('~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat');

data = test_data.X_test(1:data_size, :);
t_data = test_data.T_test(1:data_size);
w_ml = train_result.w_ml;
w0_ml = train_result.w0_ml;
w_map = train_result.w_map;
w0_map = train_result.w0_map;

cal_mtx(:, 1:2) = data(:, 1:2);
cal_mtx(:, 3:4) = data(:, 1:2).^2;
sum_mse_ml = (~data(:, 3))' * ((cal_mtx*w_ml + w0_ml - t_data).^2);
sum_mse_map = (~data(:, 3))' * ((cal_mtx*w_map + w0_map - t_data).^2);

land_num = sum(data(:,3)==0);
mse_ml = sqrt(sum_mse_ml/land_num)
mse_map = sqrt(sum_mse_map/land_num)

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/target/target_result.mat" ...
mse_ml mse_map;
