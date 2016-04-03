#! OCTAVE-INTERPRETER-NAME -qf
clear;

block_num = [14, 18];
real_size = 10000;
data_size = 10000;
block_size = 50;
data_point = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Test_data1_hw1.mat');
train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
train_result = load('~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat');

mean_x1 = train_result.mean_x1;
mean_x2 = train_result.mean_x2;
var_x1 = train_result.var_x1;
var_x2 = train_result.var_x2;
w_ml = train_result.w_ml;
w0 = train_result.w0;

x1_min = train_data.x1_bound(1);
x2_min = train_data.x2_bound(1);

acc_err = 0;
point_count = 0;

for idx=1:1:data_size
	data = data_point.X_test(idx, 1:3);
	target = data_point.T_test(idx);

	if data(3) == 0
		result = 0;
		for idm1=1:1:block_num(1)
			for idm2=1:1:block_num(2)
				addpath ~/Spring_2016/ML/2016_ML_HW1_v4/
				temp = myGaussian(data(1), data(2), mean_x1(idm1, idm2),...
				mean_x2(idm1, idm2), var_x1(idm1, idm2), var_x2(idm1, idm2));
				rmpath ~/Spring_2016/ML/2016_ML_HW1_v4/
				%accumulate the values from each models
				result += temp * w_ml((idm1-1)*block_num(2)+idm2);
			end		
		end

		%add the bias
		result += w0;
		acc_err += (result - target)^2;
		point_count += 1;
	end
end

mse_ml = sqrt(acc_err/point_count);

save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/test/test_result.mat" mse_ml
