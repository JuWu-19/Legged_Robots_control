function dy = rl_eqns(y,u)

    q = [y(1); y(2); y(3)];
    dq = [y(4); y(5); y(6)];

    M = eval_M(q);
    C = eval_C(q, dq);
    G = eval_G(q);
    B = [1 0; 0 1; -1 -1]; % write it directly to speed up

    n = 6;   
    dy = zeros(n, 1);
    dy(1) = y(4);
    dy(2) = y(5);
    dy(3) = y(6);
    dy(4:6) = M \ (-C*dq - G + B*u);

end