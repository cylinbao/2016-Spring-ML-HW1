#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [7, 9];
real_size = 10000;
data_size = 10000;
block_size = 100;
data_point = load('Test_data1_hw1.mat');
train_result = load('train_result.mat');
train_data = load('Train_data_hw1.mat');

mean_x1 = train_result.mean_x1;
mean_x2 = train_result.mean_x2;
var_x1 = train_result.var_x1;
var_x2 = train_result.var_x2;
w_ml = train_result.w_ml;

x1_min = train_data.x1_bound(1);
x2_min = train_data.x2_bound(1);

acc_err = 0;
point_count = 0;

for idx=1:1:data_size
	data = data_point.X_test(idx, 1:3);
	ans = data_point.T_test(idx);

	if data(3) == 0
		blk_idx1 = uint32((data(1) - x1_min)*20/block_size);
		blk_idx2 = uint32((data(2) - x2_min)*20/block_size);
		if blk_idx1 == 0
			blk_idx1 = 1;
		end
		if blk_idx2 == 0
			blk_idx2 = 1;
		end
		result = myGaussian(data(1), data(2), mean_x1(blk_idx1, blk_idx2),...
							mean_x2(blk_idx1, blk_idx2), var_x1(blk_idx1, blk_idx2), ...
							var_x2(blk_idx1, blk_idx2));
		result = result * w_ml((blk_idx1-1)*block_num(2)+blk_idx2);

		acc_err += (result - ans)^2;
		point_count += 1;
	end

end

mse_ml = sqrt(acc_err/point_count)

save -append -mat "test_result.mat" mse_ml
