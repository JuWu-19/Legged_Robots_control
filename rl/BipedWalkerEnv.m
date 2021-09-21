classdef BipedWalkerEnv < rl.env.MATLABEnvironment 
    
    %% Properties
    properties
        % Sample time
        Ts = 0.02
        
        % End time
        Tf = 20
        
        % Reward function
        rewardType = 'default'
        penaltyForFalling = 10
        penaltyFactorForTorque = 0.02
        totalRewardForNotFalling = 1
        constantReward % computed in constructor
        rewardForNewStep = 0
        rewardFactorForDistance = 1
        rewardFactorForSpeed = 0.1
        penaltyForTooLargeHeadAngle = 1
        penaltyForTooSmallHeadAngle = 0.1
        penaltyForTooLargeLegsAngle = 0.5
        penaltyForTooSmallStep = 1
        penaltyForTooLargeStep = 1
        
        % Constants
        envConstants = {}
    end
    
    properties
        % Initialize system state [q1, q2, q3, dq1, dq2, dq3, x_abs]'
        State = zeros(7,1)
    end
    
    properties(Access = protected)
        % Timestep counter
        timestep = 0

        % Initialize internal flag to indicate episode termination
        IsDone = false
        
        % Handle for the figure
        Figure
        
        % Counter of steps
        nbOfSteps = 0
        
        % x_abs of previous step
        previousStepPosition = 0
        
        % sln for further analysis of the simulation
        sln
        
        % Perturbations
        internalPerturbations % computed in constructor
        internalPerturbationSigma = [123e-3; 123e-3; 17e-2; 8e-1; 2.8; 1.2];
        externalPerturbation % computed in constructor
        durationExternalPerturbation = 0.2
        stepToStartExternalPerturbation = 10
        stopTimeExternalPerturbation
        externalPerturbationActive = 0
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        function this = BipedWalkerEnv(rewardType, Ts, Tf, envConstants, internalPerturbationType, externalPerturbation)
            % Initialize Observation settings
            numObs = 7;
            ObservationInfo = rlNumericSpec([numObs 1]);
            ObservationInfo.Name = 'States';
            ObservationInfo.Description = 'q1, q2, q3, dq1, dq2, dq3, x_abs';
            
            % Initialize Action settings   
            numAct = 2;
            ActionInfo = rlNumericSpec([numAct 1], 'LowerLimit', -1, 'UpperLimit', 1);
            ActionInfo.Name = 'Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            
            % Initialize property values
            this.timestep = 0;
            this.rewardType = rewardType;
            this.Ts = Ts;
            this.Tf = Tf;
            this.constantReward = this.totalRewardForNotFalling * Ts / Tf;
            this.envConstants = envConstants;
            this.sln.T = cell(envConstants.maxNbOfSteps, 1);
            this.sln.Y = cell(envConstants.maxNbOfSteps, 1);
            this.sln.u = cell(envConstants.maxNbOfSteps, 1);
            this.initializeInternalPerturbations(Tf/Ts, internalPerturbationType);
            this.externalPerturbation = externalPerturbation * envConstants.l1;
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            this.timestep = this.timestep + 1;

            % Extract useful variables
            state = this.State;
            q = state(1:3);
            dq = state(4:6);
            x_abs = state(7);
            nbSteps = state(8);
            u = this.envConstants.uMax * max(-1, min(1, Action)); % scale the action with uMax

            IsDone = 0;

            % Evolution of the system during Ts.
            % Assume constant control input during this time (except if external perturbation)
            y0 = [q, dq];
            options = odeset('RelTol',1e-5, 'Events', @event_func);
            if this.externalPerturbationActive
                [T, Y, ~, YE] = ode45(@(t, y) rl_eqns_perturbation(t, y, u, this.externalPerturbation, this.stopTimeExternalPerturbation), [0 this.Ts], y0, options);
            else
                [T, Y, ~, YE] = ode45(@(t, y) rl_eqns(y, u), [0 this.Ts], y0, options);
            end

            if ~isempty(this.sln.T{nbSteps+1})
                t_prev = this.sln.T{nbSteps+1}(end,:);
            elseif nbSteps > 0 && ~isempty(this.sln.T{nbSteps})
                t_prev = this.sln.T{nbSteps}(end,:);
            else
                t_prev = 0;
            end
            this.sln.T{nbSteps+1} = [this.sln.T{nbSteps+1}; t_prev + T];
            this.sln.Y{nbSteps+1} = [this.sln.Y{nbSteps+1}; Y];
            this.sln.u{nbSteps+1} = [this.sln.u{nbSteps+1}; repmat(u', size(Y,1), 1)];

            z_h = this.envConstants.l1*cos(Y(:,1));
            if nnz(z_h < 0.1)
%                 disp('fall');
                YE = []; % so that we do not enter the while loop
                IsDone = 1;
            end
            while ~isempty(YE) % we have a foot contact, so need to switch leg and continue the integration for the remaining time
                % Impact map
                q_m = YE(1:3)';
                dq_m = YE(4:6)';
                [q_p, dq_p] = impact(q_m, dq_m);

                % Add 1 step
                [x0, ~, ~, ~] = kin_swf(q_m);
                x_abs = x_abs + x0;
                nbSteps = nbSteps + 1;
                if nbSteps >= this.envConstants.maxNbOfSteps
                    IsDone = 1;
                    break;
                elseif this.externalPerturbationActive == 0 && nbSteps == this.stepToStartExternalPerturbation-1
                    this.externalPerturbationActive = 1;
                    this.stopTimeExternalPerturbation = T(end) + this.durationExternalPerturbation;
                end

                % Continue integration
                y0 = [q_p; dq_p];
                t0 = T(end);
                if this.externalPerturbationActive
                    [T, Y, ~, YE] = ode45(@(t, y) rl_eqns_perturbation(t, y, u, this.externalPerturbation, this.stopTimeExternalPerturbation), [t0 this.Ts], y0, options);
                else
                    [T, Y, ~, YE] = ode45(@(t, y) rl_eqns(y, u), [t0 this.Ts], y0, options);
                end
                this.sln.T{nbSteps+1} = [this.sln.T{nbSteps+1}; t_prev + T];
                this.sln.Y{nbSteps+1} = [this.sln.Y{nbSteps+1}; Y];
                this.sln.u{nbSteps+1} = [this.sln.u{nbSteps+1}; repmat(u', size(Y,1), 1)];
            end
            
            if this.externalPerturbationActive
                if T(end) > this.stopTimeExternalPerturbation
                    this.externalPerturbationActive = 0;
                else
                    this.stopTimeExternalPerturbation = this.stopTimeExternalPerturbation - T(end);
                end
            end

            q = Y(end, 1:3)';
            dq = Y(end, 4:6)';
            
            if any(mod(abs(q), pi) > pi/2) % too large angles => stop
                IsDone = 1;
            end

            % Encode the new state and observations
            Observation = [q; dq; x_abs] + this.internalPerturbations(this.timestep,:)';
            LoggedSignals.State = [q; dq; x_abs; nbSteps];
            this.State = LoggedSignals.State;
            this.IsDone = IsDone;

            % Reward
            Reward = getReward(this, Action);
            
            % use notifyEnvUpdated to signal that the environment has been updated (to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function [InitialObservation, LoggedSignals] = reset(this)
            q0 = [pi/9; -pi/9; 0];
            dq0 = [0; 0; 8];
            x_abs = 0;
            nbSteps = 0;
            InitialObservation = [q0; dq0; x_abs];
            LoggedSignals.State = [q0; dq0; x_abs; nbSteps];
            this.State = LoggedSignals.State;
            
            this.timestep = 0;
            this.nbOfSteps = 0;
            this.previousStepPosition = 0;
            this.externalPerturbationActive = 0;
            this.sln.T = cell(this.envConstants.maxNbOfSteps, 1);
            this.sln.Y = cell(this.envConstants.maxNbOfSteps, 1);
            this.sln.u = cell(this.envConstants.maxNbOfSteps, 1);
            
            % use notifyEnvUpdated to signal that the environment has been updated (to update visualization)
            notifyEnvUpdated(this);
        end
    end

    %% Additional Methods
    methods        
        % Reward function
        function Reward = getReward(this, Action)
            if strcmp(this.rewardType, 'default')
                q1 = this.State(1);
                dq1 = this.State(4);
                dx_h = dq1*this.envConstants.l1*cos(q1);
                q2 = this.State(2);
                q3 = this.State(3);
                Reward = this.rewardFactorForSpeed * dx_h + this.constantReward;
                if mod(abs(q3), pi) > 0.3*pi
                    Reward = Reward - this.penaltyForTooLargeHeadAngle;
                end
                if mod(q3, 2*pi) < pi/40 || mod(q3, 2*pi) >= pi
                    Reward = Reward - this.penaltyForTooSmallHeadAngle;
                end
                if mod(abs(q1), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if mod(abs(q2), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if this.State(8) > this.nbOfSteps % new step
                    this.nbOfSteps = this.State(8);
                    dx = this.State(7) - this.previousStepPosition;
                    this.previousStepPosition = this.State(7);

                    Reward = Reward + this.rewardFactorForDistance * dx;
                    if dx < 2 * this.envConstants.l1 * sin(pi/18)
                        Reward = Reward - this.penaltyForTooSmallStep;
                    elseif dx > 2 * this.envConstants.l1 * sin(pi/6)
                        Reward = Reward - this.penaltyForTooLargeStep;
                    end
                end
            
            elseif strcmp(this.rewardType, 'lowSpeed')
                q1 = this.State(1);
                dq1 = this.State(4);
                dx_h = dq1*this.envConstants.l1*cos(q1);
                q2 = this.State(2);
                q3 = this.State(3);
                Reward = this.constantReward;
                if dx_h <= 0.02
                    Reward = Reward - 1;
                else
%                     Reward = Reward + nthroot(dx_h, 9) * exp(-6*dx_h);
                    Reward = Reward + exp(-6*dx_h) / 7;
                end
                if mod(abs(q3), pi) > 0.2*pi
                    Reward = Reward - this.penaltyForTooLargeHeadAngle;
                end
                if mod(abs(q1), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if mod(abs(q2), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if this.State(8) > this.nbOfSteps % new step
                    this.nbOfSteps = this.State(8);
                    dx = this.State(7) - this.previousStepPosition;
                    this.previousStepPosition = this.State(7);

                    Reward = Reward + this.rewardFactorForDistance * dx;
                    if dx < 2 * this.envConstants.l1 * sin(pi/40)
                        Reward = Reward - 0.1*this.penaltyForTooSmallStep;
                    elseif dx > 2 * this.envConstants.l1 * sin(pi/6)
                        Reward = Reward - this.penaltyForTooLargeStep;
                    end
                end

            elseif strcmp(this.rewardType, 'highStepFrequency')
                q1 = this.State(1);
                dq1 = this.State(4);
                dx_h = dq1*this.envConstants.l1*cos(q1);
                q2 = this.State(2);
                q3 = this.State(3);
                Reward = this.rewardFactorForSpeed/10 * dx_h + this.constantReward;
                if mod(abs(q3), pi) > 0.3*pi
                    Reward = Reward - this.penaltyForTooLargeHeadAngle;
                end
                if mod(q3, 2*pi) < pi/40 || mod(q3, 2*pi) >= pi
                    Reward = Reward - this.penaltyForTooSmallHeadAngle;
                end
                if mod(abs(q1), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if mod(abs(q2), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if this.State(8) > this.nbOfSteps % new step
                    this.nbOfSteps = this.State(8);
                    dx = this.State(7) - this.previousStepPosition;
                    this.previousStepPosition = this.State(7);

                    Reward = Reward + this.rewardFactorForDistance/10 * dx;
                    if dx > 2 * this.envConstants.l1 * sin(pi/6)
                        Reward = Reward - this.penaltyForTooLargeStep;
                    end
                    
                    if dx > 0
                        Reward = Reward + 0.9;
                    else
                        Reward = Reward - 1;
                    end
                end
                
            elseif strcmp(this.rewardType, 'lowStepFrequency')
                this.rewardType = 'largeSteps';
                Reward = getReward(this, Action);
                
            elseif strcmp(this.rewardType, 'largeSteps')
                q1 = this.State(1);
                dq1 = this.State(4);
                dx_h = dq1*this.envConstants.l1*cos(q1);
                q3 = this.State(3);
                Reward = this.rewardFactorForSpeed/10 * dx_h + this.constantReward;
                if mod(abs(q3), pi) > 0.3*pi
                    Reward = Reward - this.penaltyForTooLargeHeadAngle;
                end
                if mod(q3, 2*pi) < pi/40 || mod(q3, 2*pi) >= pi
                    Reward = Reward - this.penaltyForTooSmallHeadAngle;
                end
                if this.State(8) > this.nbOfSteps % new step
                    this.nbOfSteps = this.State(8);
                    dx = this.State(7) - this.previousStepPosition;
                    this.previousStepPosition = this.State(7);

                    Reward = Reward + this.rewardFactorForDistance * dx * dx * dx;
                    if dx < 2 * this.envConstants.l1 * sin(pi/18)
                        Reward = Reward - this.penaltyForTooSmallStep;
                    end
                end
                
            elseif strcmp(this.rewardType, 'smallSteps')
                q1 = this.State(1);
                dq1 = this.State(4);
                dx_h = dq1*this.envConstants.l1*cos(q1);
                q2 = this.State(2);
                q3 = this.State(3);
                Reward = this.rewardFactorForSpeed/10 * dx_h + this.constantReward;
                if mod(abs(q3), pi) > 0.3*pi
                    Reward = Reward - this.penaltyForTooLargeHeadAngle;
                end
                if mod(abs(q1), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if mod(abs(q2), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if this.State(8) > this.nbOfSteps % new step
                    this.nbOfSteps = this.State(8);
                    dx = this.State(7) - this.previousStepPosition;
                    this.previousStepPosition = this.State(7);

                    Reward = Reward + this.rewardFactorForDistance * dx;
                    if dx > 2 * this.envConstants.l1 * sin(pi/6)
                        Reward = Reward - this.penaltyForTooLargeStep;
                    end
                    
                    delta = 0.02; % to tune
                    if dx <= delta
                        Reward = Reward - 1;
                    else
                        a = 5; % to tune
                        Reward = Reward + exp(-a*dx_h);
                    end
                end

            elseif strcmp(this.rewardType, 'defaultWithTorque')
                q1 = this.State(1);
                dq1 = this.State(4);
                dx_h = dq1*this.envConstants.l1*cos(q1);
                q2 = this.State(2);
                q3 = this.State(3);
                Reward = this.rewardFactorForSpeed * dx_h + this.constantReward - this.penaltyFactorForTorque * sum(Action.^2);
                if mod(abs(q3), pi) > 0.3*pi
                    Reward = Reward - this.penaltyForTooLargeHeadAngle;
                end
                if mod(q3, 2*pi) < pi/40 || mod(q3, 2*pi) >= pi
                    Reward = Reward - this.penaltyForTooSmallHeadAngle;
                end
                if mod(abs(q1), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if mod(abs(q2), pi) > pi/3
                    Reward = Reward - this.penaltyForTooLargeLegsAngle;
                end
                if this.State(8) > this.nbOfSteps % new step
                    this.nbOfSteps = this.State(8);
                    dx = this.State(7) - this.previousStepPosition;
                    this.previousStepPosition = this.State(7);

                    Reward = Reward + this.rewardFactorForDistance * dx;
                    if dx < 2 * this.envConstants.l1 * sin(pi/18)
                        Reward = Reward - this.penaltyForTooSmallStep;
                    elseif dx > 2 * this.envConstants.l1 * sin(pi/6)
                        Reward = Reward - this.penaltyForTooLargeStep;
                    end
                end

            else
                disp('Unknown reward function... use ''default''');
                this.rewardType = 'default';
                Reward = getReward(this, Action);
            end
        end
        
        % Internal perturbations
        function initializeInternalPerturbations(this, N, type)
            this.internalPerturbations = zeros(N, size(this.State,1));
            if 1 <= type && type <= length(this.internalPerturbationSigma)
                this.internalPerturbations(:,type) = this.internalPerturbationSigma(type) * randn(N, 1);
            end
        end
        
        % Visualization method
        function plot(this)
            % Initiate the visualization
            if isempty(this.Figure) || ~isvalid(this.Figure)
                this.Figure = figure('Visible', 'on', 'HandleVisibility', 'off');
                ha = gca(this.Figure);
                ha.XLimMode = 'manual';
                ha.YLimMode = 'manual';
                ha.XLim = [-3 3];
                ha.YLim = [-0.8 1.2];
                hold(ha,'on');
            end
            
            % Update the visualization
            envUpdatedCallback(this)
        end
    end
    
    methods (Access = protected)
        % update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
            if ~isempty(this.Figure) && isvalid(this.Figure)
                % Set visualization figure as the current figure
                ha = gca(this.Figure);
                cla(ha);

                q = this.State(1:3);
                x0 = this.State(7);
                l1 = this.envConstants.l1;
                l2 = this.envConstants.l2;
                l3 = this.envConstants.l3;
                q1 = q(1);
                q2 = q(2);
                q3 = q(3);

                x1 = (l1*sin(q1))/2 + x0;
                z1 = (l1*cos(q1))/2;
                x2 = l1*sin(q1) - (l2*sin(q2))/2 + x0;
                z2 = l1*cos(q1) - (l2*cos(q2))/2;
                x3 = l1*sin(q1) + (l3*sin(q3))/2 + x0;
                z3 = l1*cos(q1) + (l3*cos(q3))/2;

                x_h = l1*sin(q1) + x0;
                z_h = l1*cos(q1);

                x_t = l1*sin(q1) + l3*sin(q3) + x0;
                z_t = l1*cos(q1) + l3*cos(q3);

                x_swf = l1*sin(q1) - l2*sin(q2) + x0;
                z_swf = l1*cos(q1) - l2*cos(q2);

                lw = 2;
                % plot a line for "ground"
                plot(ha, [-1 + x_h, 1 + x_h], [0, 0], 'color', 'black');
                % links
                plot(ha, [x0, x_h], [0, z_h], 'b', 'linewidth', lw); 
                plot(ha, [x_h, x_t], [z_h, z_t], 'r', 'linewidth', lw); 
                plot(ha, [x_h, x_swf], [z_h, z_swf], 'g', 'linewidth', lw);
                % point masses
                mz = 40;
                plot(ha, x1, z1, '.b', 'markersize', mz);
                plot(ha, x2, z2, '.g', 'markersize', mz); 
                plot(ha, x3, z3, '.r', 'markersize', mz);

                % Refresh rendering in the figure window
                axis(ha, 'square');
                ha.XLim = [-1 + x_h, 1 + x_h];
                ha.YLim = [-0.8, 1.2];
                drawnow();
            end
        end
    end

    %% Getters Methods
    methods        
        function ObsInfo = getObservationInfo(this)
            ObsInfo = this.ObservationInfo;
        end
        
        function numObs = getNumObs(this)
            numObs = this.getObservationInfo().Dimension(1);
        end
        
        function ActInfo = getActionInfo(this)
            ActInfo = this.ActionInfo;
        end
        
        function numAct = getNumAct(this)
            numAct = this.getActionInfo().Dimension(1);
        end

        function sln = getSLN(this)
            sln = this.sln;
            if this.nbOfSteps < this.envConstants.maxNbOfSteps && ~isempty(this.sln.T{this.nbOfSteps+1})
                lastStep = this.nbOfSteps + 1 ;
            else
                lastStep = this.nbOfSteps;
            end
            sln.T = sln.T(1:lastStep);
            sln.Y = sln.Y(1:lastStep);
            sln.u = sln.u(1:lastStep);
        end
    end
end
