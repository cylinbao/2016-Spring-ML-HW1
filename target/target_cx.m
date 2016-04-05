#! OCTAVE-INTERPRETER-NAME -qf
clear;

data_size = 10000;
test_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Test_data1_hw1.mat');
train_result = load('~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat');

data = test_data.X_test(1:data_size, :);
t_data = test_data.T_test(1:data_size);
w_ml_cx = train_result.w_ml_cx;
w0_ml_cx = train_result.w0_ml_cx;
w_map_cx = train_result.w_map_cx;
w0_map_cx = train_result.w0_map_cx;
w_baye_cx = train_result.w_baye_cx;
w0_baye_cx = train_result.w0_baye_cx;

cal_mtx(:, 1:2) = data(:, 1:2);
cal_mtx(:, 3:4) = data(:, 1:2).^2;
sum_mse_ml = (~data(:, 3))' * ((cal_mtx*w_ml_cx + w0_ml_cx - t_data).^2);
sum_mse_map = (~data(:, 3))' * ((cal_mtx*w_map_cx + w0_map_cx - t_data).^2);
sum_mse_baye = (~data(:, 3))' * ((cal_mtx*w_baye_cx + w0_baye_cx - t_data).^2);

land_num = sum(data(:,3)==0);
mse_ml_cx = sqrt(sum_mse_ml/land_num)
mse_map_cx = sqrt(sum_mse_map/land_num)
mse_baye_cx = sqrt(sum_mse_baye/land_num)

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/target/target_result.mat" ...
mse_ml_cx mse_map_cx mse_baye_cx;
