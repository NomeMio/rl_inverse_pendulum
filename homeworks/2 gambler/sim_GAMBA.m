clear all
close all
clc

figure('Position', [100 100 1000 400])

policy_iteration_GAMBA
value_iteration_GAMBA

if ~exist('imgs', 'dir')
    mkdir('imgs')
end
exportgraphics(gcf, fullfile('imgs', 'gambler_policies.jpg'), 'Resolution', 200)
