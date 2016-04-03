#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [14, 18];
block_size = 50;
real_size = 50000;
%data_size = 50000;

total_round = 5;
round_size = real_size / total_round;
data_size = real_size - round_size;

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

train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
X_train = train_data.X_train;
T_train = train_data.T_train;
x1_bound = train_data.x1_bound;
x2_bound = train_data.x2_bound;

acc_mean_x1 = zeros(block_num(1), block_num(2));
acc_mean_x2 = zeros(block_num(1), block_num(2));
acc_var_x1 = zeros(block_num(1), block_num(2));
acc_var_x2 = zeros(block_num(1), block_num(2));
acc_w_ml = zeros(block_num(1) * block_num(2));
acc_w0 = 0;

%step = real_size/data_size;

for rnd=1:1:total_round
	if rnd == 1
		X_train_cx = X_train([round_size+1:real_size], :);
		T_train_cx = T_train([round_size+1:real_size]);
	else rnd==total_round
		X_train_cx = X_train([1:real_size-round_size], :);
		T_train_cx = T_train([1:real_size-round_size]);
	end
	end1 = (rnd-1) * (round_size);
	head2 = (rnd*round_size) + 1;
	X_train_cx = X_train([1:end1 head2:real_size],:);
	T_train_cx = T_train([1:end1 head2:real_size]);

% calculate the mean and var of each blocks
for i=1:1:data_size
	data = X_train_cx(i, 1:2);
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

acc_mean_x1 .+= mean_x1;
acc_mean_x2 .+= mean_x2;

var_x1 = (acc_x1_sq ./ count_x1) .- mean_x1.^2;
var_x2 = (acc_x2_sq ./ count_x2) .- mean_x2.^2;

acc_var_x1 .+= var_x1;
acc_var_x2 .+= var_x2;

% calculate the design matrix
for idx1=1:1:data_size
	for idx2=1:1:block_num(1)
		for idx3=1:1:block_num(2)
			data = X_train_cx(idx1, 1:2);
			addpath ~/Spring_2016/ML/2016_ML_HW1_v4/
			design_mat(idx1, (idx2-1)*block_num(2)+idx3) = myGaussian(data(1), data(2), ...
			mean_x1(idx2, idx3), mean_x2(idx2, idx3), ...
			var_x1(idx2, idx3), var_x2(idx2, idx3));
			rmpath ~/Spring_2016/ML/2016_ML_HW1_v4/
		end
	end
end

w_ml = inv(design_mat' * design_mat) * (design_mat') * (T_train_cx(1:data_size));

acc_w_ml .+= w_ml;

%below part is computing w0
acc_md = 0;
for idm1=1:1:block_num(1)
	for idm2=1:1:block_num(2)
		acc_data = 0;
		for idd=1:1:data_size
			data = X_train_cx(idd, 1:2);
			addpath ~/Spring_2016/ML/2016_ML_HW1_v4/
			temp = myGaussian(data(1), data(2), mean_x1(idm1), mean_x2(idm2), ...
							var_x1(idm1), var_x2(idm2));
			rmpath ~/Spring_2016/ML/2016_ML_HW1_v4/
			acc_data += temp;
		end
		acc_md += w_ml((idm1-1)*block_num(2)+idm2)*acc_data/data_size;
	end
end

acc_t = ones(1,data_size)*T_train_cx(1:data_size);

w0 = acc_t/data_size - acc_md;
acc_w0 += w0;

end

mean_x1 = acc_mean_x1 ./ total_round;
mean_x2 = acc_mean_x2 ./ total_round;
var_x1 = acc_var_x1 ./ total_round;
var_x2 = acc_var_x2 ./ total_round;
w_ml = acc_w_ml ./ total_round;
w0 = acc_w0 / total_round;

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result_cx.mat"...
 mean_x1 mean_x2 var_x1 var_x2 w_ml w0;
