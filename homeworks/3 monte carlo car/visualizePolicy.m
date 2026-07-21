function visualizePolicy(roadGrid, policy, valueFunction, policyName, positions, maxSpeed, N)
% Runs the greedy policy from every start cell and plots:
%   top row   : route overlay on the grid (one per start + best overall)
%   bottom row: cumulative reward curve   (one per start + best overall)
%   last two cols: action heatmap and value heatmap (v=0 slice)
% Also plots the optimal action per cell as arrows, and (if N is given)
% a heatmap of visit counts. Every figure is saved as a PNG under
% img/<policyName>/.

if nargin < 7
    N = [];
end

A = max(policy);

outDir = fullfile('img', matlab.lang.makeValidName(policyName));
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

getPositionIndex = @(y,x) find(positions(:,1) == y & positions(:,2) == x, 1);
getStateIndex    = @(state) ...
    (getPositionIndex(state(2),state(1))-1)*(2*maxSpeed+1)^2 ...
    + (state(3)+maxSpeed)*(2*maxSpeed+1) + (state(4)+maxSpeed) + 1;

%% run from every starting cell
[startRows, startCols] = find(roadGrid == 2);
nStarts    = length(startRows);
allStates  = cell(nStarts, 1);
allRewards = cell(nStarts, 1);
bestIdx    = 1;
bestLen    = Inf;

for k = 1:nStarts
    [s, r] = greedyRun([startCols(k), startRows(k)], roadGrid, policy, getStateIndex, A);
    allStates{k}  = s;
    allRewards{k} = r;
    if length(s) < bestLen
        bestLen = length(s);
        bestIdx = k;
    end
end

%% figure layout: 2 rows x (nStarts + 1 best + 2 maps) cols
nCols = nStarts + 3;
fig1 = figure('Name', ['Policy: ' policyName], 'NumberTitle', 'off');

for k = 1:nStarts
    subplot(2, nCols, k);
    plotRoute(roadGrid, allStates{k}, positions, maxSpeed);
    title(sprintf('Start (%d,%d)', startCols(k), startRows(k)), 'FontSize', 7);

    subplot(2, nCols, nCols + k);
    cumR = cumsum(allRewards{k});
    plot(1:length(cumR), cumR, 'b-', 'LineWidth', 1);
    xlabel('Step'); ylabel('Cum. reward');
    title(sprintf('Return %.1f', sum(allRewards{k})), 'FontSize', 7);
    grid on;
end

% best run column
subplot(2, nCols, nStarts + 1);
plotRoute(roadGrid, allStates{bestIdx}, positions, maxSpeed);
title('Best run', 'FontSize', 8);

subplot(2, nCols, nCols + nStarts + 1);
cumR = cumsum(allRewards{bestIdx});
plot(1:length(cumR), cumR, 'r-', 'LineWidth', 1.5);
xlabel('Step'); ylabel('Cum. reward');
title(sprintf('Best return %.1f', sum(allRewards{bestIdx})), 'FontSize', 8);
grid on;

% action heatmap (v=0 slice)
subplot(2, nCols, nStarts + 2);
actionMap = buildMap(roadGrid, positions, policy, getStateIndex);
imagesc(actionMap, 'AlphaData', ~isnan(actionMap));
set(gca, 'Color', 'k');
colormap(gca, jet(A));
clim([1 A]);
colorbar;
title('Action map (v=0)', 'FontSize', 8);
axis equal tight; xlabel('col'); ylabel('row');

subplot(2, nCols, nCols + nStarts + 2);
% (empty — keeps grid symmetrical; value map sits in last col)
axis off;

% value heatmap (v=0 slice)
subplot(2, nCols, nStarts + 3);
valueMap = buildMap(roadGrid, positions, valueFunction, getStateIndex);
imagesc(valueMap, 'AlphaData', ~isnan(valueMap));
set(gca, 'Color', 'k');
colormap(gca, parula);
colorbar;
title('Value map (v=0)', 'FontSize', 8);
axis equal tight; xlabel('col'); ylabel('row');

subplot(2, nCols, nCols + nStarts + 3);
axis off;

sgtitle(['Results — ' policyName]);
saveGraph(fig1, outDir, 'overview');

%% optimal policy arrows (v=0 slice)
fig2 = figure('Name', ['Optimal policy: ' policyName], 'NumberTitle', 'off');
plotOptimalPolicy(roadGrid, positions, policy, getStateIndex);
title(['Optimal policy — ' policyName]);
saveGraph(fig2, outDir, 'optimal_policy');

%% visit-count heatmap
if ~isempty(N)
    fig3 = figure('Name', ['Visit counts: ' policyName], 'NumberTitle', 'off');
    visitMap = buildVisitMap(roadGrid, positions, N, maxSpeed);
    imagesc(visitMap, 'AlphaData', ~isnan(visitMap));
    set(gca, 'Color', 'k');
    colormap(gca, hot);
    colorbar;
    title(['Visit counts — ' policyName]);
    axis equal tight; xlabel('col'); ylabel('row');
    saveGraph(fig3, outDir, 'visit_heatmap');
end

end

%% -----------------------------------------------------------------------
function saveGraph(fig, outDir, name)
exportgraphics(fig, fullfile(outDir, [name '.png']), 'Resolution', 150);
end

%% -----------------------------------------------------------------------
function [runStates, runRewards] = greedyRun(startXY, roadGrid, policy, getStateIndex, A)
% Try every possible first action, then follow the greedy policy.
% Keep the trajectory with the highest total return.
% This is necessary because the start state (v=0) is only ever visited
% with a random first action during training, so policy(start, v=0)
% is unreliable — but the policy works well from all subsequent states.
maxSteps = 5000;
state = [startXY(1), startXY(2), 0, 0];
bestStates  = [];
bestRewards = [];
bestScore   = -Inf;

for a0 = 1:A
    [sp, reward] = modelCar(state, a0, roadGrid);
    tStates  = [getStateIndex(state), getStateIndex(sp)];
    tRewards = reward;
    s = sp;
    while reward ~= 0 && length(tStates) < maxSteps
        [sp, reward] = modelCar(s, policy(getStateIndex(s)), roadGrid);
        tStates  = [tStates,  getStateIndex(sp)];
        tRewards = [tRewards, reward];
        s = sp;
    end
    tStates = tStates(1:end-1);
    score = sum(tRewards);
    if score > bestScore
        bestScore   = score;
        bestStates  = tStates;
        bestRewards = tRewards;
    end
end
runStates  = bestStates;
runRewards = bestRewards;
end

%% -----------------------------------------------------------------------
function plotRoute(roadGrid, runStates, positions, maxSpeed)
routeGrid = roadGrid;
for i = 1:length(runStates)
    tmp    = runStates(i) - 1;
    tmp    = floor(tmp / (2*maxSpeed+1));  % skip vy
    tmp    = floor(tmp / (2*maxSpeed+1));  % skip vx
    posIdx = tmp + 1;
    r = positions(posIdx, 1);
    c = positions(posIdx, 2);
    if routeGrid(r, c) == 1
        routeGrid(r, c) = 5;
    end
end
% 0=black, 1=grey(road), 2=green(start), 3=red(finish), 4=unused, 5=blue(route)
imagesc(routeGrid);
colormap(gca, [0 0 0; 0.55 0.55 0.55; 0 0.75 0; 0.9 0 0; 0 0 0; 0.1 0.5 1]);
clim([0 5]);
axis equal tight;
end

%% -----------------------------------------------------------------------
function map = buildMap(roadGrid, positions, dataVec, getStateIndex)
% Returns a grid where each drivable cell holds dataVec at the v=0 state.
map = NaN(size(roadGrid));
for r = 1:size(roadGrid, 1)
    for c = 1:size(roadGrid, 2)
        posIdx = find(positions(:,1)==r & positions(:,2)==c, 1);
        if ~isempty(posIdx)
            sIdx = getStateIndex([c, r, 0, 0]);
            if ~isempty(sIdx)
                map(r, c) = dataVec(sIdx);
            end
        end
    end
end
end

%% -----------------------------------------------------------------------
function map = buildVisitMap(roadGrid, positions, N, maxSpeed)
% Sums N (states x actions) over every velocity and every action that
% belongs to a given grid cell, giving total visit count per cell.
map = NaN(size(roadGrid));
nVel = (2*maxSpeed+1)^2;
for posIdx = 1:size(positions, 1)
    r = positions(posIdx, 1);
    c = positions(posIdx, 2);
    firstState = (posIdx-1)*nVel + 1;
    lastState  = posIdx*nVel;
    map(r, c) = sum(N(firstState:lastState, :), 'all');
end
end

%% -----------------------------------------------------------------------
function [dx, dy] = actionVector(a)
% Maps an action index (1-9, matching modelCar.m) to a direction arrow.
% 9 = no acceleration change (not handled by modelCar -> velocity holds).
vectors = [ 1  0;   % 1: +vx
           -1  0;   % 2: -vx
            0  1;   % 3: +vy
            0 -1;   % 4: -vy
            1  1;   % 5: +vx +vy
           -1 -1;   % 6: -vx -vy
            1 -1;   % 7: +vx -vy
           -1  1;   % 8: -vx +vy
            0  0];  % 9: no change
dx = vectors(a, 1);
dy = vectors(a, 2);
end

%% -----------------------------------------------------------------------
function plotOptimalPolicy(roadGrid, positions, policy, getStateIndex)
% Draws the best action (v=0 slice) as an arrow on every drivable cell.
imagesc(roadGrid > 0);
colormap(gca, [0 0 0; 0.8 0.8 0.8]);
axis equal tight; xlabel('col'); ylabel('row');
hold on;

X = []; Y = []; U = []; V = [];
for posIdx = 1:size(positions, 1)
    r = positions(posIdx, 1);
    c = positions(posIdx, 2);
    sIdx = getStateIndex([c, r, 0, 0]);
    if isempty(sIdx)
        continue;
    end
    a = policy(sIdx);
    [dx, dy] = actionVector(a);
    if dx == 0 && dy == 0
        plot(c, r, 'k.', 'MarkerSize', 6);
    else
        X(end+1) = c; %#ok<AGROW>
        Y(end+1) = r; %#ok<AGROW>
        U(end+1) = dx; %#ok<AGROW>
        V(end+1) = dy; %#ok<AGROW>
    end
end
quiver(X, Y, U, V, 0.4, 'r', 'LineWidth', 1);
hold off;
end
