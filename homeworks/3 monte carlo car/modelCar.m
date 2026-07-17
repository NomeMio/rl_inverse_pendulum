function [newState, reward] = modelCar(state, action, roadGrid)
    position.x = state(1, 1);
    position.y = state(1, 2);
    velocity.x = state(1, 3);
    velocity.y = state(1, 4);
    maxVelocity = 3;
    oldVelocity = [velocity.x, velocity.y];
    if(action==1)
        velocity.x = min(velocity.x + 1, maxVelocity);
    elseif(action==2)
        velocity.x = max(velocity.x - 1, -maxVelocity);
    elseif(action==3)
        velocity.y = min(velocity.y + 1, maxVelocity);
    elseif(action==4)
        velocity.y = max(velocity.y - 1, -maxVelocity);
    elseif(action==5)
        velocity.x = min(velocity.x + 1, maxVelocity);
        velocity.y = min(velocity.y + 1, maxVelocity);
    elseif(action==6)
        velocity.x = max(velocity.x - 1, -maxVelocity);
        velocity.y = max(velocity.y - 1, -maxVelocity);
    elseif(action==7)
        velocity.x = min(velocity.x + 1, maxVelocity);
        velocity.y = max(velocity.y - 1, -maxVelocity);
    elseif(action==8)   
        velocity.x = max(velocity.x - 1, -maxVelocity);
        velocity.y = min(velocity.y + 1, maxVelocity);
    end
    if velocity.x == 0 && velocity.y == 0
        velocity.x = oldVelocity(1);
        velocity.y = oldVelocity(2);
    end
    position.x = position.x + velocity.x;
    position.y = position.y + velocity.y;
    
    result = checkBoundry(roadGrid, position);
     
    reward = -1;
    if result == 0
        position = newStartPosition(roadGrid);
        velocity.x = 0;
        velocity.y = 0;
        reward = -1;
    elseif result == 3
        reward = 0;
    end



    newState = [position.x, position.y, velocity.x, velocity.y];
end 

function result = checkBoundry(road, position)
    if position.x < 1 || position.x > size(road, 2) || ...
       position.y < 1 || position.y > size(road, 1)
        result = 0;
    else
        result = road(position.y, position.x);
    end
end



function state = newState(roadGrid)
    position = newStartPosition(roadGrid);
    state = [position.x, position.y, 0, 0]; % Initialize velocity to zero
end
function position = newStartPosition(roadGrid)
[rows, cols] = find(roadGrid == 2);
randomIndex = randi(length(rows));
position.x = cols(randomIndex);
position.y = rows(randomIndex);
end
