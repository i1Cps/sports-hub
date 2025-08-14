import time
import argparse

import jax
from jax import vmap
import mujoco
import mujoco.viewer
from mujoco import mjx

from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

from .inference_registry import ENV_REGISTRY


# Inference environments
def main(env_name: str, number_of_agents: int, rng: int):
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. Choose from {list(env_name.keys())}"
        )

    # Instantiate the inference environment
    inference_env = ENV_REGISTRY[env_name](number_of_agents)

    jit_get_all_obs_func = jax.jit(inference_env.get_all_obs)
    jit_preprocess_actions_func = jax.jit(inference_env.preprocess_actions)
    check_for_termination_func = inference_env.check_for_termination

    m = inference_env.model
    d = mujoco.MjData(m)

    simulated_time = inference_env.frame_skip * m.opt.timestep

    # Create JAX and Numpy RNG
    jax_rng = jax.random.PRNGKey(rng)

    # Reset every agent 
    agents_to_reset = [True] * number_of_agents
    d, jax_rng = inference_env.reset(d, agents_to_reset, jax_rng, reset_all=True)
    mujoco.mj_forward(m, d)

    # Create agent brain
    brain_neurons = model.load_params(inference_env.path_to_brain_neurons)
    brain_input = ppo_networks.make_ppo_networks(
        inference_env.observation_dim,
        inference_env.action_dim,
        policy_hidden_layer_sizes=inference_env.brain_structure,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_brain_func = ppo_networks.make_inference_fn(brain_input)
    agent_brain = make_brain_func(brain_neurons)
    batched_agent_brains = vmap(agent_brain, in_axes=(0, 0))
    fast_batched_agent_brains = jax.jit(batched_agent_brains)

    paused = False

    def key_callback(keycode):
        if chr(keycode) == " ":
            nonlocal paused
            paused = not paused

        elif chr(keycode) == "R":
            nonlocal jax_rng, d
            reset_all = [True] * number_of_agents
            d, jax_rng = inference_env.reset(d, reset_all, jax_rng, True)
            mujoco.mj_forward(m, d)
            viewer.sync()

    @jax.jit
    def compute_control(mjx_data, rng):
        rngs = jax.random.split(rng, number_of_agents + 1)
        act_rngs = rngs[:-1]
        new_rng = rngs[-1]

        obs = jit_get_all_obs_func(mjx_data)
        actions, _ = fast_batched_agent_brains(obs, act_rngs)
        actions = jit_preprocess_actions_func(actions)

        return actions.reshape(-1), new_rng

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        viewer.sync()

        # Enable fog rendering
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1

        with viewer.lock():
            viewer.cam.lookat[:] = m.stat.center
            viewer.cam.distance = 15.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -30.0

        while viewer.is_running():
            viewer.sync()
            if not paused:
                mjx_data = mjx.put_data(m, d)
                d.ctrl, jax_rng = compute_control(mjx_data, jax_rng)

                t0 = time.perf_counter()
                for frame in range(inference_env.frame_skip):
                    mujoco.mj_step(m, d)

                    viewer.sync()

                agents_to_reset = check_for_termination_func(mjx_data)
                if any(agents_to_reset):
                    d, jax_rng = inference_env.reset(d, agents_to_reset, jax_rng)
                    mujoco.mj_forward(m, d)
                    viewer.sync()

                elapsed = time.perf_counter() - t0
                to_sleep = simulated_time - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an Inference environment")
    parser.add_argument(
        "--env",
        required=False,
        help="e.g. sprint_humanoid or longjump_humanoid",
        default="sprint_humanoid",
    )

    parser.add_argument(
        "--num_agents",
        required=False,
        help="The number of agents you want to simulate for this environment",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--rng",
        required=False,
        type=int,
        default=1,
    )
    args = parser.parse_args()
    main(args.env, args.num_agents, args.rng)
