
gridSize = 16*14;
roadGrid = zeros(16, 14);
leftBounds  = [8, 6, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1];
rightBounds = [13, 13, 13, 13, 11, 9, 8, 8, 7, 7, 7, 7, 7, 7, 6];
for row = 1:15
    roadGrid(row, leftBounds(row):rightBounds(row)) = 1;
end
roadGrid(1:4, 14) = 3;
roadGrid(16, 1:6) = 2;
imagesc(roadGrid)
colormap([0 0 0; 0.5 0.5 0.5; 0 1 0; 1 0 0])
colorbar


