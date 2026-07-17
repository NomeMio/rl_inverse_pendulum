function Vpi = iterative_policy_evaluation(P, R, pi, V0, gamma)
% compute the value function of pi

% tolerance
theta = 1e-12;

% gather dimensions
S = size(R, 1);

% determine Ppi and Rpi
Ppi = zeros(S,S);
Rpi = zeros(S,1);
for s = 1:S
    Ppi(s,:) = P(s,:,pi(s));
    Rpi(s) = R(s,pi(s));
end

while true
    % apply the Bellman update
    Vs = gamma*Ppi*V0 + Rpi;
    if norm(Vs - V0, Inf) < theta
        % interrupt evaluation
        V0 = Vs;
        break;
    else
        % iterate evaluation
        V0 = Vs;
    end
end

% assing value function
Vpi = V0;