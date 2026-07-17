clear all
close all
clc

% compare the value functions
load vstarPI.mat
VPI = V;
load vstarVI.mat
VVI = V;

disp(norm(VPI-VVI))