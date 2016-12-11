import lasagne.nonlinearities as NL
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.discriminator.mlp_discriminator import Mlp_Discriminator
from rllab.misc.instrument import stub, run_experiment_lite
import os
from rllab.sampler import parallel_sampler
parallel_sampler.initialize(n_parallel=6)

exper_spec="linearbase_trpo_0"
directory='model/'+exper_spec
if not os.path.exists(directory):
    os.makedirs(directory)


env = normalize(
        HumanoidEnv(
            vel_threshold=1,
            vel_bonus=1,
        )
    )

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(128, 64, 32)
)

base_line_optimizer = ConjugateGradientOptimizer()
baseline = GaussianMLPBaseline(env.spec,
    regressor_args={
        "mean_network": None,
        "hidden_sizes": (128, 64, 32),
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
    max_path_length=5000,
    n_itr=10001,
    discount=0.995,
    step_size=0.01,
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