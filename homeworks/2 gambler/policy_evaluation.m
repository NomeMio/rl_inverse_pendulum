function Vpi = policy_evaluation(P, R, pi, gamma)
% compute the value function of pi

% gather dimensions
S = size(R, 1);

% determine Ppi and Rpi
Ppi = zeros(S,S);
Rpi = zeros(S,1);
for s = 1:S
    Ppi(s,:) = P(s,:,pi(s));
    Rpi(s) = R(s,pi(s));
end

% solve the linear system
Vpi = (eye(S) - gamma*Ppi)\Rpi;