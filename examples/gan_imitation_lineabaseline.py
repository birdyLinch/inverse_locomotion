import lasagne.nonlinearities as NL
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
#from rllab.envs.mujoco.humanoid_env import HumanoidEnv
#from rllab.envs.mujoco.humanoid_env_origin import HumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.discriminator.mlp_discriminator import Mlp_Discriminator
from rllab.misc.instrument import stub, run_experiment_lite
import os

from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=6)

exper_spec="linearbase_gan_imitation_0"
directory='model/'+exper_spec
if not os.path.exists(directory):
    os.makedirs(directory)

disc = Mlp_Discriminator( 
        a_max=1, 
        a_min=1, 
        disc_window=2, 
        iteration=10000, 
        disc_joints_dim=20, 
        hidden_sizes=(64, 32), 
        learning_rate=3e-6,
        train_threshold=0.04,
        iter_per_train=3,
        batch_size=128,
        downsample_factor=1,
        reg=0.15,
        exper_spec=exper_spec
    )

env = normalize(
        HumanoidEnv(
            vel_deviation_cost_coeff=0,
            alive_bonus=0,
            impact_cost_coeff=0,
            disc=disc,
            vel_threshold=0.4,
            vel_bonus=0.4,
        )
    )

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(128, 64, 32)
)

baseline=LinearFeatureBaseline(env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=5000,
    n_itr=10001,
    discount=0.995,
    step_size=0.01,
    discriminator=disc,
    save_policy_every=25,
    exper_spec=exper_spec
)

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=6,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    log_dir=directory,
    # will be used
    seed=1,
    plot=True,
)
