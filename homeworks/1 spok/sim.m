% clean workspace
clear all
close all
clc
rng(3)
A = 5;


%% epsilon greedy
epsilon = 0.5; % probability of taking random action
experiment_length = 1e5;

actionValues = zeros(A, experiment_length); 
actionValues(:,1) = 0*ones(A,1);
actionNumber = zeros(A, experiment_length);

actions = zeros(1, experiment_length);
rewards = zeros(1, experiment_length);

for t = 2:experiment_length
    % choose action
    if rand < epsilon
        a = randi(A);
    else
        greedy = find(actionValues(:,t-1) == max(actionValues(:,t-1)));
        a = greedy(randi(size(greedy,1)));
    end
    actions(t) = a;
    
    r = spok(a);
    while r ==0
        r= spok(a);
    end

    rewards(t) = r;
    actionNumber(:, t) = actionNumber(:,t-1);
    actionValues(:, t) = actionValues(:, t-1);

    actionNumber(a, t) = actionNumber(a, t) + 1;
    step_size = 0.01;
    error = r - actionValues(a, t);
    actionValues(a, t) = actionValues(a, t) + step_size*error;
end

figure()
tiledlayout(3,1,"TileSpacing","compact")

ax1 = nexttile();
plot(actionNumber')
legend('1','2','3','4','5')

ax2 = nexttile();
plot(actionValues')
legend('1','2','3','4','5')


ax3 = nexttile();
plot(cumsum(rewards))
%% epsilon greedy near max
epsilon = 0.1; % probability of taking random action
experiment_length = 1e5;

actionValues = zeros(A, experiment_length); 
actionValues(:,1) = 0*ones(A,1);
actionNumber = zeros(A, experiment_length);

actions = zeros(1, experiment_length);
rewards = zeros(1, experiment_length);

for t = 2:experiment_length
    % choose action
    if rand < epsilon
        a = randi(A);
    else
        greedy = find(actionValues(:,t-1) >= (max(actionValues(:,t-1)) - 0.2) & actionValues(:,t-1) <= (max(actionValues(:,t-1)) + 0.2));
        a = greedy(randi(size(greedy,1)));
    end
    actions(t) = a;

    r = spok(a);
    while r ==0
        r= spok(a);
    end

    rewards(t) = r;
    actionNumber(:, t) = actionNumber(:,t-1);
    actionValues(:, t) = actionValues(:, t-1);

    actionNumber(a, t) = actionNumber(a, t) + 1;
    step_size = 0.01;
    error = r - actionValues(a, t);
    actionValues(a, t) = actionValues(a, t) + step_size*error;
end

figure()
tiledlayout(3,1,"TileSpacing","compact")

ax1 = nexttile();
plot(actionNumber')
legend('1','2','3','4','5')

ax2 = nexttile();
plot(actionValues')
legend('1','2','3','4','5')


ax3 = nexttile();
plot(cumsum(rewards))

%% UCB
experiment_length = 1e5;

c=100;
actionValues = zeros(A, experiment_length); 
actionValues(:,1) = 999*ones(A,1);
actionNumber = zeros(A, experiment_length);

actions = zeros(1, experiment_length);
rewards = zeros(1, experiment_length);

for t = 2:experiment_length
    % choose action
    actions_upper_confidence = actionValues(:,t-1) + c * sqrt(log(t) ./ actionNumber(:,t-1));
    pos_actions = find( actions_upper_confidence==max(actions_upper_confidence));
    a=pos_actions(randi(size(pos_actions,1)));
    actions(t) = a;

    r = spok(a);
    while r ==0
        r= spok(a);
    end

    rewards(t) = r;
    actionNumber(:, t) = actionNumber(:,t-1);
    actionValues(:, t) = actionValues(:, t-1);

    actionNumber(a, t) = actionNumber(a, t) + 1;
    step_size = 0.01;
    error = r - actionValues(a, t);
    actionValues(a, t) = actionValues(a, t) + step_size*error;
end

figure()
tiledlayout(3,1,"TileSpacing","compact")

ax1 = nexttile();
plot(actionNumber')
legend('1','2','3','4','5')

ax2 = nexttile();
plot(actionValues(:,10000:experiment_length)')
legend('1','2','3','4','5')


ax3 = nexttile();
plot(cumsum(rewards))