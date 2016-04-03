#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [14, 18];
real_size = 612226;
data_size = 612226;
block_size = 50;
prediction_ml = zeros(691, 886);
data_point = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/X_all_data.mat');
train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
train_result = load('~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat');

mean_x1 = train_result.mean_x1;
mean_x2 = train_result.mean_x2;
var_x1 = train_result.var_x1;
var_x2 = train_result.var_x2;
w_ml = train_result.w_ml;
w0 = train_result.w0;

step = uint32(real_size/data_size);
x1_min = train_data.x1_bound(1);
x2_min = train_data.x2_bound(1);

for idx=1:1:data_size
	data = data_point.X_all(idx*step, 1:3);

	result = 0;
	if data(3) == 0
		for idm1=1:1:block_num(1)
			for idm2=1:1:block_num(2)
				addpath ~/Spring_2016/ML/2016_ML_HW1_v4/
				temp = myGaussian(data(1), data(2), mean_x1(idm1, idm2), ...
								mean_x2(idm1, idm2), var_x1(idm1, idm2), var_x2(idm1, idm2));
				rmpath ~/Spring_2016/ML/2016_ML_HW1_v4/
				%accumulate the values from each model
				result += w_ml((idm1-1)*block_num(2)+idm2)*temp;
			end
		end
		%add bias to the result
		result += w0;
	end

	result_idx1 = int32((data(1)-x1_min)*20+1);
	result_idx2 = int32((data(2)-x2_min)*20+1);
	prediction_ml(result_idx1, result_idx2) = result;
end

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/test/predict_result.mat" ...
prediction_ml

clf;
colormap('default');
contour(prediction_ml);
