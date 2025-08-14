import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"os\.fork\(\) was called.*multithreaded code.*",
)

import json
import argparse
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import jax
from jax import numpy as jp
from brax import envs
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

import matplotlib.pyplot as plt
from datetime import datetime
import functools
import cv2
import imageio
import vlc
from tqdm import trange, tqdm

from .learning import ppo_train
from . import registry

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

CAMERA = "cam_default"

# Create VLC media players for training alerts.
def create_media_players() -> Dict[str, vlc.MediaPlayer]:
    return {
        "file_start": vlc.MediaPlayer("sports_hub/alerts/chop.mp3"),
        "training_start": vlc.MediaPlayer("sports_hub/alerts/drums.mp3"),
        "training_end": vlc.MediaPlayer("sports_hub/alerts/drums.mp3"),
        "inference_complete": vlc.MediaPlayer("sports_hub/alerts/chop.mp3"),
        "training_eval": vlc.MediaPlayer("sports_hub/alerts/brrping.mp3"),
    }



# Setup the PPO network factory using layer sizes from PPO params, return func partial
def setup_network_factory(ppo_train_params: Dict[str, Any]) -> Callable:
    network_factory_params = ppo_train_params.get("network_factory", {})
    # Extract hidden layers dims, use 32 * 4 if not present
    policy_hidden_layer_sizes: Tuple[int, ...] = tuple(
        network_factory_params.get("policy_hidden_layer_sizes", (32, 32, 32, 32))
    )
    # Extract hidden layers dims, use 256 * 5 if not present
    value_hidden_layer_sizes: Tuple[int, ...] = tuple(
        network_factory_params.get("value_hidden_layer_size", (256,) * 5)
    )
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
    )
    # Update ppo_train_params with the network factory
    ppo_train_params["network_factory"] = make_networks_factory
    return make_networks_factory


# Lambda function for the JAX training loop to display evaluation metrics
def progress(
    num_steps: int,
    metrics: Dict[str, float],
    agent_data: Dict[str, List[Any]],
    times: List[datetime],
    max_y: float,
    max_x: float,
    ax: plt.Axes,
    fig: plt.Figure,
) -> None:
    if num_steps == 0:
        times.append(datetime.now())
        return

    reward: float = metrics["eval/episode_reward"]
    reward_std: float = metrics["eval/episode_reward_std"]
    times.append(datetime.now())
    agent_data["x"].append(num_steps)
    agent_data["y"].append(reward)
    agent_data["err"].append(reward_std)

    print(f"Progress: {num_steps} steps, reward: {reward:.2f}")

    ax.clear()
    ax.set_xlim([0, max_x])
    ax.set_ylim([0, max_y])
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    ax.set_title(f"reward: {reward:.3f}")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.errorbar(agent_data["x"], agent_data["y"], yerr=agent_data["err"])

    fig.canvas.draw()
    fig.canvas.flush_events()


def main(config_data, env_name, rng):
    print("Hello, Welcome to Sports Hub")

    alerts: Dict[str, vlc.MediaPlayer] = create_media_players()
    alerts["file_start"].play()

    print("Environment name:", env_name)
    env = envs.get_environment(env_name)

    video_dir_path = os.path.join("videos", env_name)

    use_eval_env = config_data.get("eval_env", False)
    eval_env = envs.get_environment(env_name + "_eval") if use_eval_env else env

    render_width = env.render_width
    render_height = env.render_height
    print(f"Environment Render Shape: {render_width} x {render_height}")

    fps = float(1.0 / env.dt)
    print(f"fps: {fps:.2f}")

    print(
        f"\nAgent Observations size: {env.observation_size} Agent Actions size: {env.action_size}"
    )

    # Get ppo hyperparameters object
    ppo_train_params = config_data["ppo_train_params"]
    # Add users rng value (If stated)
    ppo_train_params["seed"] += rng 

    # Record training time stamps
    times = [datetime.now()]

    # Split Random keys
    key = jax.random.PRNGKey(rng)
    preview_key, training_key, evaluation_key = jax.random.split(key, 3)

    # JIT compile reset and step functions for performance
    env_jit_reset = jax.jit(env.reset)
    env_jit_step = jax.jit(env.step)
    eval_env_jit_reset = jax.jit(eval_env.reset)
    eval_env_jit_step = jax.jit(eval_env.step)

    # NOTE: Preview Environment to visualise what we are training -----------------------------------------------------------------
    if config_data.get("preview", True):
        print("Previewing The environment")
        num_preview_rollouts: int = config_data["num_preview_rollouts"]
        preview_rollout_frames: List[Any] = []

        for _ in range(num_preview_rollouts):
            state = env_jit_reset(preview_key)
            rollout = [state.pipeline_state]

            # Create a trajectory of 500 steps
            for _ in range(500):
                ctrl = -0.1 * jp.ones(env.sys.nu)
                state = env_jit_step(state, ctrl)
                rollout.append(state.pipeline_state)

            # Generate frames from the rollout so we can preview
            frames = env.render(
                rollout, height=render_height, width=render_width, camera=CAMERA
            )
            preview_rollout_frames.extend(frames)

        # Display the frames using OpenCV
        for frame in preview_rollout_frames:
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Preview Rollout", frame_bgr)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        print("Preview finished")

    # NOTE: Metrics ----------------------------------------------------------------------------

    # Data containers for plotting progress
    agent_data: Dict[str, List[Any]] = {
        "x": [0],
        "y": [0],
        "err": [0],
    }

    max_y: float = config_data["max_y"]
    max_x: float = ppo_train_params["num_timesteps"] * 1.25
    plt.ion()  # Turn on interactive plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a progress update function (wrapper around our helper)
    progress_fn = lambda steps, met: progress(
        steps,
        met,
        agent_data,
        times,
        max_y,
        max_x,
        ax,
        fig,
    )

    times = [datetime.now()]

    # NOTE: Training ------------------------------------------------------------------------

    number_of_steps = ppo_train_params["num_timesteps"]

    # This initialises the network sizes and adds them to the user hyperparameters, use the return in evaluation when reconstructing network
    make_network_factory = setup_network_factory(ppo_train_params)

    train_fn = functools.partial(ppo_train.train, **ppo_train_params)

    print(f"\nStarting {number_of_steps} steps of training:\n")
    alerts["training_start"].play()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress_fn, seed=rng)

    print("\nEnding the training...")
    print(f"\ntime to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    # Save model parameters
    model_dir_path = os.path.join("agent_brains", env_name, "model")
    os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
    model.save_params(model_dir_path, params)

    # NOTE: Evaluation ---------------------------------------------------------------------------

    print("\nStarting the evaluation")
    alerts["training_eval"].play()

    times = [datetime.now()]

    # Construct the agents brain
    agent_ppo_network = make_network_factory(
        eval_env.observation_size,
        eval_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    agent_make_inference_func = ppo_networks.make_inference_fn(agent_ppo_network)
    agent_params = model.load_params(model_dir_path)
    agent_inference_fn = agent_make_inference_func(agent_params)
    jit_agent_inference_fn = jax.jit(agent_inference_fn)

    num_inference_rollouts = config_data["num_inference_rollouts"]
    num_inference_steps = config_data.get(
        "num_inference_steps", ppo_train_params["episode_length"]
    )

    # Certain environments that use more than 4000 timesteps will crash system due to running out of RAM, 4000*50 1080p frames  
    render_inference_per_frame = config_data.get("render_inference_per_frame", False)

    # Prepare to collect frames for the GIF
    gif_frames = []
    video_dir_path = os.path.join("videos", env_name)
    os.makedirs(video_dir_path, exist_ok=True)
    video_path = os.path.join(video_dir_path, "inference_rollouts.mp4")
    gif_path = os.path.join(video_dir_path, "inference_rollouts.gif")

    print("generating the rollouts")

    if render_inference_per_frame:
        with imageio.get_writer(video_path, fps=fps) as writer:
            rollout_number = 0
            for rollout_idx in range(num_inference_rollouts):
                state = eval_env_jit_reset(evaluation_key)
                rollout = [state.pipeline_state]
                steps_this_rollout = 0
                rollout_reward = 0.0
                descaled_frames = []

                # Create a trajectory
                for step in trange(num_inference_steps, colour="green"):
                    act_rng, evaluation_key = jax.random.split(evaluation_key)
                    ctrl, _ = jit_agent_inference_fn(state.obs, act_rng)
                    state = eval_env_jit_step(state, ctrl)
                    steps_this_rollout += 1
                    rollout_reward += state.reward

                    # single-frame render
                    single_frame = eval_env.render(
                        [state.pipeline_state],
                        height=render_height,
                        width=render_width,
                        camera=CAMERA,
                    )[0]
                    writer.append_data(single_frame)

                    # Scale down the frames for gifs
                    descaled_frame = eval_env.render(
                        [state.pipeline_state],
                        height=225,
                        width=400,
                        camera=CAMERA,
                    )[0]

                    # Stack the frames from each rollout
                    descaled_frames.append(descaled_frame)

                    if state.done:
                        break

                gif_frames.append(descaled_frames)
                print(
                    f"rollout: {rollout_number}, total steps: {steps_this_rollout}, reward: {rollout_reward}"
                )

                rollout_number = rollout_number + 1
                alerts["inference_complete"].play()

    else:
        with imageio.get_writer(video_path, fps=fps) as writer:
            rollout_number = 0
            for _ in range(num_inference_rollouts):
                state = eval_env_jit_reset(evaluation_key)
                rollout = [state.pipeline_state]
                steps_this_rollout = 0
                rollout_reward = 0.0

                # Create a trajectory
                for step in trange(num_inference_steps, colour="green"):
                    act_rng, evaluation_key = jax.random.split(evaluation_key)
                    ctrl, _ = jit_agent_inference_fn(state.obs, act_rng)
                    state = eval_env_jit_step(state, ctrl)
                    rollout.append(state.pipeline_state)
                    steps_this_rollout += 1
                    rollout_reward += state.reward

                    if state.done:
                        break

                print("calling env.render()")
                # Collect frames from each rollout
                frames = eval_env.render(
                    rollout,
                    height=render_height,
                    width=render_width,
                    camera=CAMERA,
                )

                descaled_frames = eval_env.render(
                    rollout,
                    height=225,
                    width=400,
                    camera=CAMERA,
                )

                # Write the frames to the same video file
                for frame in frames:
                    writer.append_data(frame)

                # Stack the frames from each rollout
                gif_frames.append(descaled_frames)

                print(
                    f"rollout: {rollout_number}, total steps: {steps_this_rollout}, reward: {rollout_reward}"
                )
                rollout_number = rollout_number + 1
                alerts["inference_complete"].play()

    # Save the GIF using the collected frames
    print("Saving GIF to", gif_path)
    for i, stack_of_frames in enumerate(gif_frames):
        new_gif_path = gif_path[:-4] + str(i + 1) + gif_path[-4:]
        imageio.mimsave(new_gif_path, stack_of_frames, fps=fps, loop=0)
    flattened_frames = [
        frame for stack_of_frames in gif_frames for frame in stack_of_frames
    ]
    imageio.mimsave(gif_path, flattened_frames, fps=fps, loop=0)

    print(f"GIF saved to {gif_path}")

    # After training and before closing the figures, save the last progress plot.
    plot_dir_path = os.path.join("plots", env_name)
    os.makedirs(plot_dir_path, exist_ok=True)
    plot_path = os.path.join(plot_dir_path, "rewards.png")
    fig.savefig(plot_path)
    print(f"Training progress plot saved to {plot_path}")

    # Clean up
    plt.ioff()
    plt.close("all")

    alerts["training_end"].play()
    time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Agent")
    parser.add_argument(
        "--environment",
        required=False,
        help="e.g. fetch_humanoid or stairs_quadruped",
        default="sprint_humanoid",
    )

    parser.add_argument(
        "--rng",
        required=False,
        type=int,
        default=0,
    )
    args = parser.parse_args()
    env = args.environment

    # Load config
    with open("sports_hub/environments/" + env + "/hyperparams.json") as f:
        config_data = json.load(f)
    main(config_data, args.environment, args.rng)
