#! OCTAVE-INTERPRETER-NAME -qf
clear;

data_size = 612226;
data_point = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/X_all_data.mat');
train_result = load('~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat');

data = data_point.X_all(1:data_size, :);
w_ml = train_result.w_ml;
w0_ml = train_result.w0_ml;
w_map = train_result.w_map;
w0_map = train_result.w0_map;

cal_mat(:, 1:2) = data(:, 1:2);
cal_mat(:, 3:4) = data(:, 1:2).^2;
result_ml = (cal_mat*w_ml + w0_ml) .* (~data(:, 3)) ;
result_map = (cal_mat*w_map + w0_map) .* (~data(:, 3)) ;

% form the result to output matrix format
prediction_ml_mat = flipud(reshape(result_ml, 886, 691)');
prediction_map_mat = flipud(reshape(result_map, 886, 691)');

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/target/predict_result.mat" ...
prediction_ml_mat prediction_map_mat;

colormap('default');
contour(prediction_ml_mat);
title("ML approach");
xlabel("Longitude");
ylabel("Latitude");
contour(prediction_map_mat);
title("MAP approach");
xlabel("Longitude");
ylabel("Latitude");
