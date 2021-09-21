function objective_value = optimization_fun(parameters)

    % extract parameters q0, dq0 and x
    q0 = parameters(1:3);
    dq0 = parameters(4:6);
    x = parameters(7:end); % x = [kp1; kp2; kd1; kd2; alpha]

    % run simulation
    num_steps = 10; % the higher the better, but slow
    sln = solve_eqns(q0, dq0, num_steps, x);
    results = analyse(sln, x, false);

    % calculate metrics such as distance, mean velocity and cost of transport
%     max_actuation = 30;
%     effort = ...;
%     distance = ...;
%     velocity = ...;
%     CoT = ...;
    desiredSpeed = 0.1;
    w1 = 1;
    w2 = 10;
    objective_value = w1 * abs(desiredSpeed - results.averageSpeed) + w2 * results.CoT;

    % handle corner case when model walks backwards (e.g., objective_value =
    % 1000)
    if nnz(results.stepSpeed < 0)
        objective_value = 1000;
        disp('backward')
    end

    % handle case when model falls (e.g., objective_value = 1000)
    if nnz(results.hipPosition(:,2) < 0.1) % consider fall if hip position is lower than 10cm
        objective_value = 1000;
        disp('fall')
    end
end

