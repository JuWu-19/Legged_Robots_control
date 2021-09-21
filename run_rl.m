%% Clear
% clear;

%% Parameters
% Reward
rewardType = 'default';

% Perturbations
internalPerturbationType = 0; % 1...6 means q1,q2,...,dq3. 0 means no perturbations
externalPerturbation = 0; % horizontal force applied to the hip (expressed in Newton)

% Training and simulation
maxNbOfSteps = 200;
Ts = 0.01;
Tf = 20;
maxEpisodes = 10000;
maxSteps = floor(Tf/Ts);
simOpts = rlSimulationOptions('MaxSteps', maxSteps);

% Agent and actions
AgentSelection = 'TD3';
uMax = 30;

%% Path
addpath('./kinematics', './control', './dynamics', './set_parameters/', './solve_eqns/', './visualize', './analysis', './optimization', './rl');

%% Constants
[m1, m2, m3, l1, l2, l3, g] = set_parameters();
envConstants.m1 = m1;
envConstants.m2 = m2;
envConstants.m3 = m3;
envConstants.l1 = l1;
envConstants.l2 = l2;
envConstants.l3 = l3;
envConstants.g  = g;
envConstants.uMax = uMax;
envConstants.maxNbOfSteps = maxNbOfSteps;

%% Create environment
env = BipedWalkerEnv(rewardType, Ts, Tf, envConstants, internalPerturbationType, externalPerturbation);
numObs = env.getNumObs;
numAct = env.getNumAct;
ObservationInfo = env.getObservationInfo;
ActionInfo = env.getActionInfo;

%% Agent
% Be sure to not accidentally overwrite the last agent...
if exist('agent', 'var') && ~isempty(agent)
    disp('Are you sure you want to train a new agent?')
    disp('This will delete the previous one!');
    disp('Make a copy and/or delete the previous agent before continuing');
    pause;
    disp('Sure?');
    pause;
    disp('Really sure?');
    pause;
    disp('Last chance...');
    pause;
end
switch AgentSelection
    case 'DDPG'
        agent = createDDPGAgent(numObs, ObservationInfo, numAct, ActionInfo, Ts);
    case 'TD3'
        agent = createTD3Agent(numObs, ObservationInfo, numAct, ActionInfo, Ts);
    otherwise
        disp('Enter DDPG or TD3 for AgentSelection')
end

%% Training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxEpisodes,...
    'MaxStepsPerEpisode',maxSteps,...
    'ScoreAveragingWindowLength',250,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeCount',...
    'StopTrainingValue',maxEpisodes,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',50,...
    'SaveAgentDirectory', "savedAgents_" + regexprep(datestr(now), ' |:', '-') + "/");

%% Train
trainingStats = train(agent,env,trainOpts);

%% Simulate
plot(env);
sim(env, agent, simOpts);

%% Analyse the results
sln = env.getSLN;
results = analyse(sln);
