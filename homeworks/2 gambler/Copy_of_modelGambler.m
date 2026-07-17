clear all
close all
clc

gamma = 0.9;
p = 0.5; % fair coin

% States: index s represents $(s-1) dollars
%   s=1   -> $0   (broke, absorbing)
%   s=101 -> $100 (goal, absorbing)
S = 101;

% Actions: index a represents a bet of $a (bets $1 to $99)
A = 99;

P = zeros(S, S, A);
R = zeros(S, A);

for s = 1:S
    money = s - 1; % dollar amount: 0..100

    if money == 0 || money == 100
        % absorbing states: self-loop for all actions, zero reward
        for a = 1:A
            P(s, s, a) = 1;
        end
        continue
    end

    for a = 1:A
        bet = a;
        max_bet = min(money, 100 - money);

        if bet > max_bet
            % invalid bet for this state: self-loop, zero reward
            P(s, s, a) = 1;
        else
            s_win  = money + bet + 1; % index of state after winning
            s_lose = money - bet + 1; % index of state after losing

            P(s, s_win,  a) = p;
            P(s, s_lose, a) = 1 - p;

            % reward +1 on reaching $100, -1 on going broke
            r_win  =  double(money + bet == 100);
            r_lose = -double(money - bet == 0);
            R(s, a) = p * r_win + (1 - p) * r_lose;
        end
    end
end

% sanity check: each row of P must sum to 1 for every action
for a = 1:A
    err = max(abs(sum(P(:,:,a), 2) - 1));
    if err > 1e-10
        fprintf('Action %d: row-sum error = %g\n', a, err);
    end
end
fprintf('Sanity check done\n');

save('modelGambler.mat', 'P', 'R', 'gamma');
