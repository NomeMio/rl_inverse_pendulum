function r = spok(a)

% 1 sasso 2 carta 3 forbice 4 lizard 5 spok

r=0;
avv=randi(5);

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