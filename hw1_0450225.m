#! OCTAVE-INTERPRETER-NAME -qf
clear;

addpath ./train
train_ml;
rmpath ./train
addpath ./test
test_ml;
prediction_ml;
rmpath ./test
