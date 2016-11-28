import lasagne.nonlinearities as NL
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.discriminator.mlp_discriminator import Mlp_Discriminator

from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=3)

disc = Mlp_Discriminator( 
        a_max=0.8, 
        a_min=0.8, 
        disc_window=4, 
        iteration=10000, 
        disc_joints_dim=10, 
        hidden_sizes=(128, 64, 32), 
        learning_rate=1e-4,
        train_threshold=0.25,
        iter_per_train=1,
        batch_size=512
    )

env = normalize(
        HumanoidEnv(
            vel_deviation_cost_coeff=0,
            alive_bonus=0.2,
            ctrl_cost_coeff=0,
            impact_cost_coeff=0,
            disc=disc,
            vel_threshold=0.8,
            vel_bonus=0.8
        )
    )

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(100, 50, 25)
)



base_line_optimizer = ConjugateGradientOptimizer()
baseline = GaussianMLPBaseline(env.spec,
    regressor_args={
        "mean_network": None,
        "hidden_sizes": (100, 50, 25),
        "hidden_nonlinearity": NL.tanh,
        "optimizer": base_line_optimizer,
        "use_trust_region": True,
        "step_size": 0.01,
        "learn_std": True,
        "init_std": 1.0,
        "adaptive_std": False,
        "std_share_network": False,
        "std_hidden_sizes": (32, 32),
        "std_nonlinearity": None,
        "normalize_inputs": True,
        "normalize_outputs": True,
    })

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=10000,
    n_itr=10000,
    discount=0.995,
    step_size=0.01,
    discriminator=disc,
    save_policy_every=25,
    exper_spec="model4|"
)

algo.train()
