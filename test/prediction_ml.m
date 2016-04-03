#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [7, 9];
real_size = 612226;
data_size = 612226;
block_size = 100;
prediction_ml = zeros(691, 886);
data_point = load('../data/X_all_data.mat');
train_data = load('../data/Train_data_hw1.mat');
train_result = load('../train/train_result.mat');

mean_x1 = train_result.mean_x1;
mean_x2 = train_result.mean_x2;
var_x1 = train_result.var_x1;
var_x2 = train_result.var_x2;
w_ml = train_result.w_ml;

step = uint32(real_size/data_size);
x1_min = train_data.x1_bound(1);
x2_min = train_data.x2_bound(1);

for idx=1:1:data_size
	data = data_point.X_all(idx*step, 1:3);

	if data(3) == 0
		blk_idx1 = uint32((data(1) - x1_min)*20/block_size);
		blk_idx2 = uint32((data(2) - x2_min)*20/block_size);
		if blk_idx1 == 0
			blk_idx1 = 1;
		end
		if blk_idx2 == 0
			blk_idx2 = 1;
		end
		addpath ../
		result = myGaussian(data(1), data(2), mean_x1(blk_idx1, blk_idx2),...
							mean_x2(blk_idx1, blk_idx2), var_x1(blk_idx1, blk_idx2), ...
							var_x2(blk_idx1, blk_idx2));
		rmpath ../
		result = result * w_ml((blk_idx1-1)*block_num(2)+blk_idx2);
	else
		result = 0;
	end

	result_idx1 = int32((data(1)-x1_min)*20+1);
	result_idx2 = int32((data(2)-x2_min)*20+1);
	prediction_ml(result_idx1, result_idx2) = result;
end

save -append -mat "predict_result.mat" prediction_ml

clf;
colormap('default');
contour(prediction_ml)
