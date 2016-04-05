#! OCTAVE-INTERPRETER-NAME -qf
clear;

real_size = 50000;

train_data = load('~/Spring_2016/ML/2016_ML_HW1_v4/data/Train_data_hw1.mat');
data = train_data.X_train;
T_train = train_data.T_train;
x1_bound = train_data.x1_bound;
x2_bound = train_data.x2_bound;

round_total = 10;
round_size = real_size / round_total;
data_size = real_size - round_size;

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

design_mat(:, 1:2) = data_cx(:, 1:2);
design_mat(:, 3:4) = data_cx(:, 1:2).^2;

% using ml appraoch 
w_ml(:, end+1) = inv(design_mat' * design_mat) * (design_mat') * (T_train_cx);
w0_ml(end+1) = mean(T_train_cx) - (mean(design_mat) * w_ml(:, rnd));

% using map appraoch 
lambda = -91.3;
reg_mtx = lambda * diag(ones(1,4));
w_map(:, end+1) = inv(reg_mtx + design_mat'*design_mat)*(design_mat')*(T_train_cx);
w0_map(end+1) = mean(T_train_cx) - (mean(design_mat) * w_map(:, rnd));

% using bayesian approach
pkg load statistics
alpha_num = -30;
beta_num = 0.3;

S_mtx = inv(alpha_num*diag(ones(1,4)) + beta_num*design_mat'*design_mat);
M_mtx = beta_num * S_mtx * design_mat' * T_train_cx;

w_baye(:, end+1) = (mvnrnd(M_mtx, S_mtx))';
w0_baye(end+1) = mean(T_train) - (mean(design_mat) * w_baye(:, rnd));
pkg unload statistics

end % end of the for-loop

w_ml_cx = mean(w_ml')';
w0_ml_cx = mean(w0_ml);
w_map_cx = mean(w_map')';
w0_map_cx = mean(w0_map);
w_baye_cx = mean(w_baye')';
w0_baye_cx = mean(w0_baye);

% save matrix into file train_result
save -append -mat "~/Spring_2016/ML/2016_ML_HW1_v4/train/train_result.mat" ...
w_ml_cx w0_ml_cx w_map_cx w0_map_cx w_baye_cx w0_baye_cx;
