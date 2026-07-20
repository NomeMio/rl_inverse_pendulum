function r = spok(a, legit)

% 1 sasso 2 carta 3 forbice 4 lizard 5 spok

if nargin < 2
    legit = true;
end

r=0;

if legit
    avv=randi(5);
else
    draw = randi(10);
    if draw >= 5
        avv = 5;
    else
        avv = draw;
    end
end

if avv == a
    return
end


if a == 1
    if avv == 3 || avv == 4
        r=1;
    else
        r=-1;
    end
    return
end
if a == 2
    if avv == 1 || avv == 5
        r=1;
    else
        r=-1;
    end
    return
end
if a == 3
    if avv == 2 || avv == 4
        r=1;
    else
        r=-1;
    end
    return
end
if a == 4
    if avv ==   5 || avv ==  2
        r=1;
    else
        r=-1;
    end
    return
end
if a == 5
    if avv == 1|| avv == 3
        r=1;
    else
        r=-1;
    end
    return
end
