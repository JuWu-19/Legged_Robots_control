## Course project of Legged robots MICRO-507

Reinforcement learning control of biped robot \
<img src="https://user-images.githubusercontent.com/58901415/160092906-153364a5-b8cd-45d5-abfe-1aff8cfc6a1d.gif" width="32%" height="32%" />
### 1. Modelling

On the basis of biped kinematics, dynamics and conservation of angular momentum, the impact map is computed to model robot step switching \
<img src="https://user-images.githubusercontent.com/58901415/160013564-1a41a63e-55ae-474a-aec0-fb17f315e1fd.png" width="46%" height="46%" />

### 2. PD feedback controller

PD controller for this under-actuated system is implemented \
<img src="https://user-images.githubusercontent.com/58901415/160087875-934062ad-d885-4c78-9151-5269e89dabd0.png" width="16%" height="16%" /> \
And then the gait metrics are introduced and the unconstrained
optimization is carried out to decide optimal parameters as\
<img src="https://user-images.githubusercontent.com/58901415/160088459-2a10b1c2-dd3d-4619-af2b-927630f49acc.png" width="28%" height="28%" />

### 3. TD3 agent control

The reinforcement learning toolbox is used to implement the environment and the agent. The reward function is designed as \
<img src="https://user-images.githubusercontent.com/58901415/160090132-19b42a7f-08e1-455b-aa0c-5b43b684c537.png" width="48%" height="48%" /> \
The episode resward during training as\
<img src="https://user-images.githubusercontent.com/58901415/160090596-9fb5abbb-c5a2-4958-8578-eaeeb63aec00.png" width="48%" height="48%" />   
It can be seen that: \
Pros: can achieve better performances
than any other controller thanks to learning \
Cons: difficult to find good reward function,
training is very slow \
Bonus: pretty robust against perturbations
