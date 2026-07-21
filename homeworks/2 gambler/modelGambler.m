clear all
close all
clc

gamma = 0.999;
p = 0.40; % fair coin

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
        for a = 1:A
            P(s, s, a) = 1;
        end
        continue
    end

    for a = 1:A
        bet = a;

        if bet > money
            P(s, s, a) = 1;
        else
            win_money  = min(money + bet, 100); % cap at $100
            lose_money = money - bet;

            s_win  = win_money  + 1;
            s_lose = lose_money + 1;

            P(s, s_win,  a) = p;
            P(s, s_lose, a) = 1 - p;

            r_win  =  double(win_money  == 100);
            r_lose = -double(lose_money == 0);
            R(s, a) = p * r_win + (1 - p) * r_lose;
        end
    end
end

for a = 1:A
    err = max(abs(sum(P(:,:,a), 2) - 1));
    if err > 1e-10
        fprintf('Action %d: row-sum error = %g\n', a, err);
    end
end
fprintf('Sanity check done\n');

save('modelGambler.mat', 'P', 'R', 'gamma');
