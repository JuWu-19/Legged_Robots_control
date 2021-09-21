function u = control(t, q, dq, q0, dq0, step_number, parameters)
    % You may call control_hyper_parameters and desired_outputs in this function
    % you don't necessarily need to use all the inputs to this control function

    if nargin < 7 || isempty(parameters)
        parameters = control_hyper_parameters(step_number);
    end

    % extract parameters
    kp1 = parameters(1);
    kp2 = parameters(2);
    kd1 = parameters(3);
    kd2 = parameters(4);
    alpha = parameters(5);

    % controller for torso
    u1 = kp1 * (q(3) - alpha) + kd1 * dq(3);

    % controller for legs
    u2 = kp2 * (-q(2) - q(1)) + kd2 * (-dq(2) - dq(1)); 

    % saturate the output torque
    u = [u1; u2];
    u = max(min(u, 30), -30);

end