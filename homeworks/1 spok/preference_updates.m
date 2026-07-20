%% Preference updates

experiment_length = 1e6;

preferences = zeros(A, experiment_length);
preferences(:,1) = 0*ones(A,1);
actionNumber = zeros(A, experiment_length);
actions = zeros(1, experiment_length);
total_reward = 0;
% probability and action selection functions that accept a preference column vector
p = @(pref) exp(pref) ./ sum(exp(pref));
pick_action = @(pref) find(rand < cumsum(p(pref)), 1);

for t = 2:experiment_length
    % choose action using probabilities computed from preferences at t-1
    probs = p(preferences(:,t-1));
    a = pick_action(preferences(:,t-1));
    actions(t) = a;

    legit_t = legit && (t <= legit_switch_at);
    r = spok(a, legit_t);
    while r == 0
        r = spok(a, legit_t);
    end

    total_reward = total_reward + r;
    actionNumber(:, t) = actionNumber(:,t-1);
    actionNumber(a, t) = actionNumber(a, t) + 1;

    % update preferences using baseline = mean reward so far
    baseline = total_reward / t;
    for ai = 1:A
        step_size = 0.0001 ;%1/max(1,actionNumber(ai,t));

        if ai == a
            preferences(ai, t) = preferences(ai, t-1) + step_size * (r - baseline) * (1 - probs(ai));
        else
            preferences(ai, t) = preferences(ai, t-1) - step_size * (r - baseline) * probs(ai);
        end
    end
end

% trim arrays
preferences = preferences(:,1:experiment_length);
actionNumber = actionNumber(:,1:experiment_length);
actions = actions(1:experiment_length);

subplot(1,3,3)
plot(preferences')
legend('1','2','3','4','5')
title('Preference updates')
