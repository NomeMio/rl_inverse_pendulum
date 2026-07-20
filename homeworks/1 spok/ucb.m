%% UCB
experiment_length = 1e6;

c=10;
actionValues = zeros(A, experiment_length);
actionValues(:,1) = 1*ones(A,1);
actionNumber = zeros(A, experiment_length);

actions = zeros(1, experiment_length);
rewards = zeros(1, experiment_length);

for t = 2:experiment_length
    % choose action
    actions_upper_confidence = actionValues(:,t-1) + c * sqrt(log(t) ./ max(actionNumber(:,t-1),1));
    pos_actions = find(actions_upper_confidence==max(actions_upper_confidence));
    a=pos_actions(randi(numel(pos_actions)));
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
    step_size =0.0001; %1/actionNumber(a, t);
    error = r - actionValues(a, t);
    actionValues(a, t) = actionValues(a, t) + step_size*error;
end

% trim arrays
actionValues = actionValues(:,1:experiment_length);
actionNumber = actionNumber(:,1:experiment_length);
actions = actions(1:experiment_length);
rewards = rewards(1:experiment_length);

subplot(1,3,2)
plot(actionValues')
legend('1','2','3','4','5')
title('UCB')
