#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [7, 9];
block_size = 100;
real_size = 50000;
data_size = 50000;
design_mat = zeros(data_size, block_num(1)*block_num(2));
acc_x1 = zeros(block_num(1), block_num(2));
acc_x2 = zeros(block_num(1), block_num(2));
acc_x1_sq = zeros(block_num(1), block_num(2));
acc_x2_sq = zeros(block_num(1), block_num(2));
count_x1 = zeros(block_num(1), block_num(2));
count_x2 = zeros(block_num(1), block_num(2));
mean_x1 = zeros(block_num(1), block_num(2));
mean_x2 = zeros(block_num(1), block_num(2));
var_x1 = zeros(block_num(1), block_num(2));
var_x2 = zeros(block_num(1), block_num(2));

train_data = load('Train_data_hw1.mat');
X_train = train_data.X_train;
T_train = train_data.T_train;
x1_bound = train_data.x1_bound;
x2_bound = train_data.x2_bound;

step = real_size/data_size;
% calculate the mean and var of each blocks
for i=1:1:data_size
	data = X_train(i*step, 1:2);
	index1 = uint32((data(1) - x1_bound(1))*20/block_size);
	index2 = uint32((data(2) - x2_bound(1))*20/block_size);

	if index1 == 0
		index1 = 1;
	end
	if index2 == 0
		index2 = 1;
	end
	
	count_x1(index1, index2) += 1;
	count_x2(index1, index2) += 1;

	acc_x1(index1, index2) += data(1);
	acc_x2(index1, index2) += data(2);

	acc_x1_sq(index1, index2) += data(1)^2;
	acc_x2_sq(index1, index2) += data(2)^2;
end

mean_x1 = acc_x1 ./ count_x1;
mean_x2 = acc_x2 ./ count_x2;

var_x1 = (acc_x1_sq ./ count_x1) .- mean_x1.^2;
var_x2 = (acc_x2_sq ./ count_x2) .- mean_x2.^2;

% calculate the design matrix
for idx1=1:1:data_size
	for idx2=1:1:block_num(1)
		for idx3=1:1:block_num(2)
			data = X_train(idx1*step, 1:2);
			design_mat(idx1, (idx2-1)*block_num(2)+idx3) = myGaussian(data(1), data(2), ...
			mean_x1(idx2, idx3), mean_x2(idx2, idx3), ...
			var_x1(idx2, idx3), var_x2(idx2, idx3));
		end
	end
end

w_ml = inv(design_mat' * design_mat) * (design_mat') * (T_train(1:data_size));

save -append -mat "train_result.mat" mean_x1 mean_x2 var_x1 var_x2 w_ml
