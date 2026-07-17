function pip = policy_improvement(P, R, V, gamma)

% gather dimensions
S = size(R, 1);
A = size(R, 2);

% improved policy
pip = zeros(S,1);

% construct the quality function
for s = 1:S
    q = zeros(A,1);
    for a = 1:A
        q(a) = R(s,a) + gamma*P(s,:,a)*V;
    end
    pip(s) = find(q == max(q), 1);
end