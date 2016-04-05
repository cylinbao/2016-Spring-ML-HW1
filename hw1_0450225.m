#! OCTAVE-INTERPRETER-NAME -qf
clear;

addpath ./train
train;
rmpath ./train
addpath ./target
prediction;
target;
rmpath ./target

clear;
