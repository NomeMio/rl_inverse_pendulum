function [learned_policy, value, Q, C] = offPolicyMC(roadGrid, positions, S, A, maxSpeed, toll, gamma)

Q = -10 * ones(S,A);   % pessimistic init: below any achievable return (-1/(1-gamma) ~ -2.5)
C = zeros(S,A);
learned_policy = ones(S,1);
for s = 1:S
    learned_policy(s) = find(Q(s,:) == max(Q(s,:)), 1, 'first');
end
value_previous = -ones(S,1);

% behavior policy b: uniform random, b(a|s) = 1/A for all s,a
% target policy pi: deterministic greedy (learned_policy)
% importance ratio when action matches target: pi(a|s)/b(a|s) = 1/(1/A) = A
% when action doesn't match target: pi(a|s)=0 -> break

getPositionIndex = @(y,x) find(positions(:,1) == y & positions(:,2) == x);
getStateIndex = @(state) (getPositionIndex(state(2),state(1))-1)*(2*maxSpeed+1)^2 + (state(3)+maxSpeed)*(2*maxSpeed+1) + (state(4)+maxSpeed) + 1;

while true
    counter = 0;
    while counter < 1000
        display(['Episode ', num2str(counter), ' of Off-Policy MC.']);
        counter = counter + 1;
        [rows, cols] = find(roadGrid == 2);
        randomIndex = randi(length(rows));
        position.x = cols(randomIndex);
        position.y = rows(randomIndex);
        state   = [position.x, position.y, 0, 0];
        actions = randi(A);   % uniform behavior for first step
        [newState, reward] = modelCar(state, actions, roadGrid);
        states  = [getStateIndex(state), getStateIndex(newState)];
        rewards = reward;
        maxSteps = 100000;
        s = newState;
        while reward ~= 0 && length(states) < maxSteps
            a = randi(A);   % uniform behavior policy
            [sp, reward] = modelCar(s, a, roadGrid);
            states  = [states,  getStateIndex(sp)];
            actions = [actions, a];
            rewards = [rewards, reward];
            s = sp;
            if reward == 0
                display(['Episode ', num2str(counter), ' terminated after ', num2str(length(states)), ' steps.']);
            end
        end
        states = states(1:end-1);
        T = length(states);
        G = 0;
        W = 1;
        for t = T:-1:1
            G = gamma*G + rewards(t);
            C(states(t), actions(t)) = C(states(t), actions(t)) + W;
            Q(states(t), actions(t)) = Q(states(t), actions(t)) + W/C(states(t), actions(t)) * (G - Q(states(t), actions(t)));
            learned_policy(states(t)) = find(Q(states(t),:) == max(Q(states(t),:)), 1, 'first');
            if actions(t) ~= learned_policy(states(t))
                break;   % target policy gives zero prob to this action -> stop
            end
            W = W * A;   % pi(a|s)/b(a|s) = 1/(1/A) = A
        end
    end
    display(['Off-Policy MC completed after ', num2str(counter), ' episodes.']);

    value = zeros(S,1);
    for s = 1:S
        value(s) = Q(s, learned_policy(s));
    end

    policy_p = learned_policy;
    for s = 1:S
        if any(C(s,:) > 0)
            policy_p(s) = find(Q(s,:) == max(Q(s,:)), 1, 'first');
        end
    end
    if norm(value - value_previous, Inf) < toll
        break;
    else
        disp(norm(value - value_previous, Inf));
        disp(norm(policy_p - learned_policy, 1));
    end

    learned_policy = policy_p;
    value_previous = value;

end

end
