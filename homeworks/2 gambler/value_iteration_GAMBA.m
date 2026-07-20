clear all
close all
clc

% for reproducibility fix random seed
rng(2)

% load model
load modelGambler.mat;

% treshold for stopping
theta = 1e-6;

% gather dimensions
S = size(R, 1);
A = size(R, 2);

% initialize V and pi
V = randn(S,1);

% count iterations
count = 0;

% repeat until convergence
while true
    % increment the counter
    count = count + 1;
    
    % maximum increment
    Delta = 0;
    for s = 1:S
        % sweep for evaluation for each action
        q = zeros(A,1);
        for a = 1:A
            q(a) = R(s,a) + gamma*P(s,:,a)*V;
        end
        % one step of improvement
        Vp = max(q);
        Delta = max([Delta, Vp - V(s)]);
        % update the estimate
        V(s) = Vp;
    end

    % interruption criterion
    if Delta < theta
        break;
    end

    %disp([count,Delta]);
end

% the optimal policy is greedy with respect to Vstar
pi = policy_improvement(P, R, V, gamma);

save vstarVI.mat V

%% plot the optimal policy
capital = (0:S-1)';

% compute Q(s,a) for contour plot, NaN for invalid bets
Q = NaN(S, A);
for s = 1:S
    money = s - 1;
    max_bet = min(money, A);
    for a = 1:max_bet
        Q(s, a) = R(s, a) + gamma * P(s, :, a) * V;
    end
end

figure()
contourf(capital, 1:A, Q', 20)
colorbar
hold on
plot(capital, pi, 'r-', 'LineWidth', 2)
xlabel('Capital ($)')
ylabel('Bet ($)')
title('Q-function and optimal policy (red)')
xlim([0 100])
ylim([1 A])

figure()
surf(capital, (1:A)', Q', 'EdgeColor', 'none')
xlabel('Capital ($)')
ylabel('Bet ($)')
zlabel('Q-value')
title('Q-function surface')
colorbar
