#! OCTAVE-INTERPRETER-NAME -qf
clear;

data_size = 612226;
data_point = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/X_all_data.mat');
train_result = load('~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat');

data = data_point.X_all(1:data_size, :);
mean_x1 = train_result.mean_x1;
mean_x2 = train_result.mean_x2;
var_x1 = train_result.var_x1;
var_x2 = train_result.var_x2;
w_map = train_result.w_map;
w0_map = train_result.w0_map;

result = (exp(-((data(:, 1) - mean_x1).^2./(2*var_x1)) ...
					-((data(:, 2) - mean_x2).^2./(2*var_x2)))*w_map + w0_map) .* (~data(:, 3));

prediction_map_mat = flipud(reshape(result, 886, 691)');

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/test/predict_result.mat" ...
prediction_map_mat;

clf;
colormap('default');
contour(prediction_map_mat);
