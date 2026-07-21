
%%

clear all
close all
clc
%%
gridSize = 16*14;
roadGrid = zeros(16, 14);
leftBounds  = [8, 6, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1];
rightBounds = [13, 13, 13, 13, 11, 9, 8, 8, 7, 7, 7, 7, 7, 7, 6];
for row = 1:15
    roadGrid(row, leftBounds(row):rightBounds(row)) = 1;
end
roadGrid(1:4, 14) = 3;
roadGrid(16, 1:6) = 2;
%finding all the positions of the road and saving the cordinates in a vector
[rows, cols] = find(roadGrid > 0);
positions = [rows, cols];
%gettin first postion as example
sizePostions=size(positions,1);


minVisits = 100;
toll = 1e-3;
alpha = 1e-2;
epsilon = 0.1;
gamma = 0.9;
maxSpeed = 3;
S = sizePostions*(2*maxSpeed+1)^2;

A = 9;

%% on-policy Monte Carlo control
[policy, value, Q, N] = onPolicyMC(roadGrid, positions, S, A, maxSpeed, minVisits, toll, epsilon, gamma);

display('Monte Carlo completed. Final policy and value function obtained.');

%% Visualize
on_policy_policy = policy;
visualizePolicy(roadGrid, policy, value, 'On-Policy MC', positions, maxSpeed, N);


%% off-policy Monte Carlo control
[learned_policy, value, Q, C, N] = offPolicyMC(roadGrid, positions, S, A, maxSpeed, toll, gamma);

%% Display learened policy
display('Monte Carlo completed. Final policy and value function obtained for off policy.');
visualizePolicy(roadGrid, learned_policy, value, 'Off-Policy MC', positions, maxSpeed, N);
