% for reproducibility fix random seed
rng(2)
modelGambler
% load model
load modelGambler.mat;

% gather dimensions
S = size(R, 1);
A = size(R, 2);

% initialize V and pi
V = randn(S,1);
pi = randi(A, [S, 1]);

% count iterations
count = 0;

% repeat until convergence
while true
    % increment the counter
    count = count + 1;

    % evaluate the current policy
    V = policy_evaluation(P, R, pi, gamma);
    % V = iterative_policy_evaluation(P, R, pi, V, gamma);

    % update the policy
    pip = policy_improvement(P, R, V, gamma);

    % interrupt if the policy is stable
    if norm(pip - pi) == 0
        break
    else
        pi = pip; % update the policy
    end
end

save vstarPI.mat V



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

subplot(1,2,1)
contourf(capital, 1:A, Q', 20)
colorbar
hold on
plot(capital, pi, 'r-', 'LineWidth', 2)
xlabel('Capital ($)')
ylabel('Bet ($)')
title('Policy iteration')
xlim([0 100])
ylim([1 A])
