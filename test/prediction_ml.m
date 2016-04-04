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
w_ml = train_result.w_ml;
w0 = train_result.w0;

result = (exp(-((data(:, 1) - mean_x1).^2./(2*var_x1)) ...
					-((data(:, 2) - mean_x2).^2./(2*var_x2)))*w_ml + w0) .* (~data(:, 3));

prediction_ml_mat = flipud(reshape(result, 886, 691)');

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/test/predict_result.mat" ...
prediction_ml_mat;

clf;
colormap('default');
contour(prediction_ml_mat);
