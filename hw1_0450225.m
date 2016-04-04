#! OCTAVE-INTERPRETER-NAME -qf
clear;

addpath ./train
train_ml;
rmpath ./train
addpath ./test
prediction_ml;
test_ml;
rmpath ./test
