#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [14, 18];
block_size = 50;
real_size = 50000;
round_total = 10;
round_size = real_size / round_total;
data_size = real_size - round_size;
lambda = 15;

train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
data = train_data.X_train(1:real_size, :);
T_train = train_data.T_train(1:real_size, :);
x1_bound = train_data.x1_bound;
x2_bound = train_data.x2_bound;

for rnd=1:1:round_total
	if rnd == 1
		data_cx = data(round_size+1:real_size, :);
		T_train_cx = T_train(round_size+1:real_size, :);
	elseif rnd == round_total
		data_cx = data(1:real_size-round_size, :);
		T_train_cx = T_train(1:real_size-round_size, :);
	else
		data_cx = data([1:round_size*(rnd-1) round_size*rnd+1:real_size], :);
		T_train_cx = T_train([1:round_size*(rnd-1) round_size*rnd+1:real_size], :);
	end

idx1_vec = ceil((data_cx(:, 1) .- (x1_bound(1)-0.01))*20/block_size);
idx2_vec = ceil((data_cx(:, 2) .- (x2_bound(1)-0.01))*20/block_size);
count_x1 = zeros(1, block_num(1) * block_num(2));
count_x2 = zeros(1, block_num(1) * block_num(2));
label_x1 = zeros(1, block_num(1) * block_num(2));
label_x2 = zeros(1, block_num(1) * block_num(2));
% label the data by the idx of the block it belong to
for idx=1:1:data_size
	idx1 = idx1_vec(idx);	
	idx2 = idx2_vec(idx);	

	label_x1(end+1, ((idx1-1)*block_num(2)+idx2)) = data_cx(idx, 1);
	label_x2(end+1, ((idx1-1)*block_num(2)+idx2)) = data_cx(idx, 2);

	count_x1((idx1-1)*block_num(2)+idx2) += 1;
	count_x2((idx1-1)*block_num(2)+idx2) += 1;
end

label_x1 = label_x1(2:end, :);
label_x2 = label_x2(2:end, :);

% calculate the mean and var of each blocks
mean_x1(end+1, 1:block_num(1)*block_num(2)) = sum(label_x1) ./ count_x1;
mean_x2(end+1, 1:block_num(1)*block_num(2)) = sum(label_x2) ./ count_x2;

var_x1(end+1, 1:block_num(1)*block_num(2)) = ...
	(sum(label_x1.^2) ./ count_x1) .- (mean_x1.^2);
var_x2(end+1, 1:block_num(1)*block_num(2)) = ...
	(sum(label_x2.^2) ./ count_x2) .- (mean_x2.^2);

% calculate the design matrix
design_mat = exp(-((data_cx(:, 1) - mean_x1(rnd, :)).^2./(2*var_x1(rnd, :))) ...
								-((data_cx(:, 2) - mean_x2(rnd, :)).^2./(2*var_x2(rnd, :))));

% calculate the w for each models and the w0
diagmtx = diag(ones(1, block_num(1)*block_num(2)));
w_map(:, end+1) = inv(lambda*diagmtx + design_mat'*design_mat)*(design_mat')*(T_train_cx);
w0(end+1) = mean(T_train_cx) - (mean(design_mat) * w_map(:, rnd));

end

% calculate the mean value of each arguments
mean_x1_cx = mean(mean_x1);
mean_x2_cx = mean(mean_x2);
var_x1_cx = mean(var_x1);
var_x2_cx = mean(var_x2);
w_map_cx = mean(w_map);
w0_map_cx = mean(w0);

% save matrix into file train_result
save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat" ...
mean_x1_cx mean_x2_cx var_x1_cx var_x2_cx w_map_cx w0_map_cx;
