#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [14, 18];
block_size = 50;
data_size = 50000;

train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
data = train_data.X_train(1:data_size, :);
T_train = train_data.T_train(1:data_size, :);
x1_bound = train_data.x1_bound;
x2_bound = train_data.x2_bound;

idx1_vec = ceil((data(:, 1) .- (x1_bound(1)-0.01))*20/block_size);
idx2_vec = ceil((data(:, 2) .- (x2_bound(1)-0.01))*20/block_size);
count_x1 = zeros(1, block_num(1) * block_num(2));
count_x2 = zeros(1, block_num(1) * block_num(2));
% label the data by the idx of the block it belong to
for idx=1:1:data_size
	idx1 = idx1_vec(idx);	
	idx2 = idx2_vec(idx);	

	label_x1(end+1, ((idx1-1)*block_num(2)+idx2)) = data(idx, 1);
	label_x2(end+1, ((idx1-1)*block_num(2)+idx2)) = data(idx, 2);

	count_x1((idx1-1)*block_num(2)+idx2) += 1;
	count_x2((idx1-1)*block_num(2)+idx2) += 1;
end

% calculate the mean and var of each blocks
mean_x1 = sum(label_x1) ./ count_x1;
mean_x2 = sum(label_x2) ./ count_x2;

var_x1 = (sum(label_x1.^2) ./ count_x1) .- (mean_x1.^2);
var_x2 = (sum(label_x2.^2) ./ count_x2) .- (mean_x2.^2);

% calculate the design matrix
design_mat = exp(-((data(:, 1) - mean_x1).^2./(2*var_x1)) ...
								-((data(:, 2) - mean_x2).^2./(2*var_x2)));

% calculate the w for each models and the w0
w_ml = inv(design_mat' * design_mat) * (design_mat') * (T_train);
w0 = mean(T_train) - (mean(design_mat) * w_ml);

% save matrix into file train_result
save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat" ...
mean_x1 mean_x2 var_x1 var_x2 w_ml w0;
