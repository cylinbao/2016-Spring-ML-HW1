#! OCTAVE-INTERPRETER-NAME -qf
clear;

addpath ./train
train_ml;
rmpath ./train
addpath ./test
prediction_ml;
test_ml;
rmpath ./test

clear;

addpath ./train
train_map;
rmpath ./train
addpath ./test
prediction_map;
test_map;
rmpath ./test

clear;

addpath ./train
train_ml_cx;
rmpath ./train
addpath ./test
prediction_ml_cx;
test_ml_cx;
rmpath ./test

clear;

addpath ./train
train_map_cx;
rmpath ./train
addpath ./test
prediction_map_cx;
test_map_cx;
rmpath ./test
