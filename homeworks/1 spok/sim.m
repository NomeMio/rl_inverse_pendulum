% clean workspace
clear all
close all
clc
rng(3)
A = 5;
legit = false; % set to false to make spok's draw a rigged (non-legit) one
legit_switch_at = 5*1e5; % iteration after which the draw becomes rigged (Inf = never switches)
fig_title = 'Late unfair game'; % set to a string to title the combined figure
img_filename = 'late_unfair_game.jpg'; % set to a filename to save the combined figure into imgs/ (empty to skip saving)

figure('Position', [100 100 1500 400])

epsilon_greedy
ucb
preference_updates

if ~isempty(fig_title)
    sgtitle(fig_title)
end

if ~isempty(img_filename)
    if ~exist('imgs', 'dir')
        mkdir('imgs')
    end
    exportgraphics(gcf, fullfile('imgs', img_filename), 'Resolution', 200)
end

