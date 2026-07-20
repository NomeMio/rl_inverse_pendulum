%% epsilon greedy mean samples (for loop; plot only action value function)
epsilon = 0.1; % probability of taking random action
max_experiment_length = 1e6;

actionValues = zeros(A, max_experiment_length);
actionValues(:,1) = 0*ones(A,1);
actionNumber = zeros(A, max_experiment_length);

actions = zeros(1, max_experiment_length);
rewards = zeros(1, max_experiment_length);

for t = 2:max_experiment_length
    % choose action
    if rand < epsilon
        a = randi(A);
    else
        greedy = find(actionValues(:,t-1) == max(actionValues(:,t-1)));
        a = greedy(randi(numel(greedy)));
    end
    actions(t) = a;

    legit_t = legit && (t <= legit_switch_at);
    r = spok(a, legit_t);
    while r == 0
        r = spok(a, legit_t);
    end

    rewards(t) = r;
    actionNumber(:, t) = actionNumber(:,t-1);
    actionValues(:, t) = actionValues(:, t-1);

    actionNumber(a, t) = actionNumber(a, t) + 1;
    step_size = 1/actionNumber(a, t);
    error = r - actionValues(a, t);
    actionValues(a, t) = actionValues(a, t) + step_size*error;
end

% trim arrays to actual length max_experiment_length (no early stopping)
actionValues = actionValues(:,1:max_experiment_length);
actionNumber = actionNumber(:,1:max_experiment_length);
actions = actions(1:max_experiment_length);
rewards = rewards(1:max_experiment_length);

subplot(1,3,1)
plot(actionValues')
legend('1','2','3','4','5')
title('Epsilon-greedy')
