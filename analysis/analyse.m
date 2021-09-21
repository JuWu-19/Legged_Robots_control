function results = analyse(sln, parameters, to_plot)
    if nargin < 3; to_plot = 1; end

    results = {};
    % calculate gait quality metrics (distance, step frequency, step length, velocity, etc.)
    num_steps = length(sln.T);
    r0 = [0; 0];
    results.stepLength = zeros(1, num_steps);
    results.stepTime = zeros(1, num_steps);
    results.stepSpeed = zeros(1, num_steps);
    for j = 1:num_steps
        [x0, ~, ~, ~] = kin_swf(sln.Y{j}(end, 1:3));
        r0 = r0 + [x0; 0];
        results.stepLength(j) = x0;
        results.stepTime(j) = (sln.T{j}(end) - sln.T{j}(1));
        results.stepSpeed(j) = results.stepLength(j) / results.stepTime(j);
    end
    results.distance = r0(1);
    results.finalTime = sln.T{end}(end);
    results.averageSpeed = results.distance / results.finalTime;
    results.averageStepLength = mean(results.stepLength);
    results.stepFrequency = num_steps / results.finalTime;
    results.nbSteps = num_steps;
    

    % calculate actuation
    u = zeros(0, 2);
    hipPos = zeros(0, 2);
    hipVel = zeros(0, 2);
    t = [];
    y = zeros(0, 6);
    r0 =[0; 0];
    for j = 1:num_steps
        Y = sln.Y{j};
        T = sln.T{j};
        [N, ~] = size(Y);
        q0 = Y(1, 1:3);
        dq0 = Y(1, 4:6);
        for i = 1:N
            q = Y(i, 1:3);
            dq = Y(i, 4:6);
            if isfield(sln, 'u') % for sln coming from our RL agent
%                 disp(['N = ', num2str(N), ', j = ', num2str(j), ', i = ', num2str(i)])
                u(end+1, :) = sln.u{j}(i,:);
            else
                u(end+1, :) = control(T(i), q, dq, q0, dq0, i, parameters)';
            end
            [x_h, z_h, dx_h, dz_h] = kin_hip(q, dq);
            hipPos = [hipPos; r0(1) + x_h, r0(2) + z_h];
            hipVel = [hipVel; dx_h, dz_h];
        end
        t = [t; T];
        y = [y; Y];
        [x0, ~, ~, ~] = kin_swf(Y(end, 1:3));
        r0 = r0 + [x0; 0];
    end
    results.torque = u;
    results.hipPosition = hipPos;
    results.hipSpeed = hipVel;
    results.time = t;
    results.q = y(:, 1:3);
    results.dq = y(:, 4:6);
    
    % effort
    results.effort = mean(sum(results.torque.^2, 2)) / (2 * 30); % Umax = 30
    
    % Cost of Transport
    results.CoT = results.effort / results.distance;

    if to_plot
        set(groot, 'DefaultAxesFontSize', 16);
        set(groot, 'DefaultLineLineWidth', 1.5);
        % plot the angles
        figure(10); clf; hold on;
        plot(results.time, results.q);
        title('Angles');
        xlabel('Time [s]');
        ylabel('Angle [rad]');
        legend('q1', 'q2', 'q3');

        % plot the hip position
        figure(11); clf; hold on;
        plot(results.hipPosition(:,1), results.hipPosition(:,2));
        title('Hip position');
        xlabel('x-coordinate [m]');
        ylabel('z-coordinate [m]');

        % plot instantaneous and average velocity
        figure(12); clf; hold on;
        plot(results.time, results.hipSpeed);
        plot(results.time, cumsum(results.hipSpeed)./(1:length(results.time))');
        plot(results.stepTime(1) + cumsum(results.stepTime(2:end)), results.stepSpeed(2:end));
        title('Hip velocity');
        xlabel('Time [s]');
        ylabel('Hip velocity [m/s]');
        legend('x-velocity', 'z-velocity', 'mean x-velocity', 'mean z-velocity', 'mean x-velocity by step');

        % plot projections of the limit cycle
        figure(13); clf; hold;
        plot(results.q, results.dq);
        title('Limit cycle');
        xlabel('q [rad]');
        ylabel('dq [rad/s]');
        legend('q1', 'q2', 'q3');

        % plot actuation
        figure(14); clf; hold on;
        plot(results.time, results.torque);
        title('Actuation');
        xlabel('Time [s]');
        ylabel('Torque [Nm]');
        legend('u1', 'u2');
    end

end