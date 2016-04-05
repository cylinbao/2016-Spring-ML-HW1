#! OCTAVE-INTERPRETER-NAME -qf
clear;

% normal trainning process
addpath ./train
train;
rmpath ./train
addpath ./target
prediction;
target;
rmpath ./target

% cross validation with k=10 trainning process
addpath ./train
train_cx;
rmpath ./train
addpath ./target
prediction_cx;
target_cx;
rmpath ./target

clear;
