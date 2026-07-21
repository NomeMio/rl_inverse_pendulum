function [policy, value, Q, N] = onPolicyMC(roadGrid, positions, S, A, maxSpeed, minVisits, toll, epsilon, gamma)

Q = zeros(S,A);
policy = randi(A,[S,1]);
policy_c = zeros(S,A);
for s = 1:S
    policy_c(s,:) = 1/A;
end

value_previous = -ones(S,1);

getPositionIndex = @(y,x) find(positions(:,1) == y & positions(:,2) == x);
getStateIndex = @(state) (getPositionIndex(state(2),state(1))-1)*(2*maxSpeed+1)^2 + (state(3)+maxSpeed)*(2*maxSpeed+1) + (state(4)+maxSpeed) + 1;

N = zeros(S,A);
iterations = 0;
while true
    iterations = iterations + 1;
    if mod(iterations, 100) == 0
        disp(['Iteration: ', num2str(iterations)]);
    end
    [rows, cols] = find(roadGrid == 2);
    randomIndex = randi(length(rows));
    position.x = cols(randomIndex);
    position.y = rows(randomIndex);
    actions = randi(A);
    state = [position.x, position.y, 0, 0]; % Initialize velocity to zero
    [newState, reward] = modelCar(state, actions, roadGrid);
    states = [getStateIndex(state), getStateIndex(newState)];
    rewards = reward;
    s = newState;
    maxSteps = 100000;
    visitedStates = false(S, 1);
    while reward ~= 0 && length(states) < maxSteps %until termination
        if rand < epsilon
            a = randi(A);
        else
            a = getGreedyAction(policy_c, getStateIndex(s)); % follow policy pi
        end
        [sp, reward] = modelCar(s, a, roadGrid);
        states  = [states,  getStateIndex(sp)];
        actions = [actions, a];
        rewards = [rewards, reward];
        s = sp; 
    end
    states = states(1:end-1);
    T = length(states);
    G = 0;
    for t = T:-1:1
        G = gamma*G + rewards(t);
        if ~visitedStates(states(t))
            visitedStates(states(t)) = true;
            N(states(t),actions(t)) = N(states(t),actions(t)) + 1;
            Q(states(t),actions(t)) = Q(states(t),actions(t)) + 1/N(states(t),actions(t))*(G-Q(states(t),actions(t)));
            A_star = find(Q(states(t),:) == max(Q(states(t),:)), 1, 'first');
            for a = 1:A
                if a == A_star
                    policy_c(states(t),a) = 1-epsilon+epsilon/A;
                else
                    policy_c(states(t),a) = epsilon/A;
                end
            end
            [~, policy(states(t))] = max(policy_c(states(t),:));
        end
    end
    value = zeros(S,1);
    for s = 1:S
        value(s) = Q(s,policy(s));
    end
    policy_p = policy; % keep current policy for unvisited states
    for s = 1:S
        if any(N(s,:) > 0)
            policy_p(s) = find(Q(s,:) == max(Q(s,:)), 1, 'first');
        end
    end
    if norm(value - value_previous, Inf) < toll && iterations > 1000
        display(['Converged after ', num2str(iterations), ' iterations.', ' Norm difference: ', num2str(norm(value - value_previous, Inf))]);
        break;
    end
    policy = policy_p;
    value_previous = value;
end

end

function greedyAction = getGreedyAction(policy, stateIndex)
    randomValue = rand;
    cumulativeProbability = cumsum(policy(stateIndex, :));
    greedyAction = find(randomValue <= cumulativeProbability, 1, 'first');
    if isempty(greedyAction)
        greedyAction = randi(size(policy, 2)); % Fallback to a random action
    end
end
