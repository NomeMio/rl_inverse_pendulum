
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
toll = 8e-2;
alpha = 1e-2;
epsilon = 0.01;
gamma = 0.60;
maxSpeed = 3;
S = sizePostions*(2*maxSpeed+1)^2; 

A = 9;

Q = zeros(S,A);

policy = randi(A,[S,1]);

policy_c= zeros(S,A);
%make the policy a equidistribuited
for s = 1:S
    policy_c(s,:) = 1/A;
end

function greedyAction = getGreedyAction(policy, stateIndex)
    randomValue = rand;
    cumulativeProbability = cumsum(policy(stateIndex, :));
    greedyAction = find(randomValue <= cumulativeProbability, 1, 'first');
    if isempty(greedyAction)
        greedyAction = randi(size(policy, 2)); % Fallback to a random action
    end

end

% initialize value_previous
value_previous = -ones(S,1);

%% start montecarlo analysis
getPositionIndex = @(y,x) find(positions(:,1) == y & positions(:,2) == x);
getStateIndex = @(state) (getPositionIndex(state(2),state(1))-1)*(2*maxSpeed+1)^2 + (state(3)+maxSpeed)*(2*maxSpeed+1) + (state(4)+maxSpeed) + 1;



N=zeros(S,A);
while true
    counter=0;
    while counter<1000
        display(['Episode ', num2str(counter), ' of Monte Carlo iteration.', ' Minimum visits: ', num2str(min(min(N))), ' of ', num2str(minVisits)]);
        counter=counter+1;
        [rows, cols] = find(roadGrid == 2);
        randomIndex = randi(length(rows));
        position.x = cols(randomIndex);
        position.y = rows(randomIndex);
        actions =randi(A);
        state = [position.x, position.y, 0, 0]; % Initialize velocity to zero
        [newState, reward] = modelCar(state, actions,roadGrid);
        states=[getStateIndex(state), getStateIndex(newState)];
        rewards=reward;
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
            s = sp; % update state  
            if reward == 0 
                display(['Episode ', num2str(counter), ' terminated after ', num2str(length(states)), ' steps.']);
            end    
        end
        states=states(1:end-1);
        T=length(states);
        G=0;
        for t=T:-1:1
            G=gamma*G+rewards(t);
            if ~visitedStates(states(t))
                visitedStates(states(t)) = true;
                N(states(t),actions(t))=N(states(t),actions(t))+1;
                Q(states(t),actions(t))=Q(states(t),actions(t))+1/N(states(t),actions(t))*(G-Q(states(t),actions(t)));
                A_star=find(Q(states(t),:) == max(Q(states(t),:)), 1,'first');
                for a=1:A
                    if a==A_star
                        policy_c(states(t),a)=1-epsilon+epsilon/A;
                    else
                        policy_c(states(t),a)=epsilon/A;
                    end
                end
                [~, policy(states(t))] = max(policy_c(states(t),:));
            end
        end
    
    end
    display(['Monte Carlo iteration completed after ', num2str(counter), ' episodes.']);

    value=zeros(S,1);
    for s=1:S
        value(s)=Q(s,policy(s));
    end

    policy_p = policy; % keep current policy for unvisited states
    for s = 1:S
        if any(N(s,:) > 0)
            policy_p(s) = find(Q(s,:) == max(Q(s,:)), 1,'first');
        end
    end
    if norm(value - value_previous, Inf) < toll
        break;
    else
        disp(norm(value - value_previous, Inf));
        disp(norm(policy_p - policy, 1));
    end

    policy = policy_p;
    value_previous = value;

end


display('Monte Carlo completed. Final policy and value function obtained.');

%% Visualize
on_policy_policy = policy;
visualizePolicy(roadGrid, policy, value, 'On-Policy MC', positions, maxSpeed);


%% OFF policy

Q = -10 * ones(S,A);   % pessimistic init: below any achievable return (-1/(1-gamma) ≈ -2.5)
C = zeros(S,A);
learned_policy = ones(S,1);
for s = 1:S
    learned_policy(s) = find(Q(s,:) == max(Q(s,:)), 1, 'first');
end
value_previous = -ones(S,1);

% behavior policy b: uniform random, b(a|s) = 1/A for all s,a
% target policy pi: deterministic greedy (learned_policy)
% importance ratio when action matches target: pi(a|s)/b(a|s) = 1/(1/A) = A
% when action doesn't match target: pi(a|s)=0 → break

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
                break;   % target policy gives zero prob to this action → stop
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
%% Display learened policy 
display('Monte Carlo completed. Final policy and value function obtained for off policy.');
visualizePolicy(roadGrid, learned_policy, value, 'Off-Policy MC', positions, maxSpeed);

