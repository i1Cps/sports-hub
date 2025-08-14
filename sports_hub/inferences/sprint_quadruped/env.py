import mujoco
from jax import numpy as jp
from mujoco import mjx
import jax
import random
from typing import Any, Tuple, List
import re

# Inference environment for 'Sprint Quadruped'
class SprintQuadruped_Inference:
    def __init__(self, number_of_agents=1, debug=False):
        scene_path = "sports_hub/inferences/sprint_quadruped/base_model.xml"

        # Original spec to copy from
        original_spec = mujoco.MjSpec.from_file(scene_path)

        # Main Spec that we are buliding on
        spec = mujoco.MjSpec.from_file(scene_path)

        # Remove the original quadruped body form the base model
        spec.delete(spec.body("quadruped"))

        # Create a dummy frame in the main spec and use it to attach each new quadruped too
        dummy_frame = spec.worldbody.add_frame()

        # Pre compute colours for each quadruped
        colours = []

        for i in range(number_of_agents):
            r = random.random()
            g = random.random()
            b = random.random()
            colours.append([r, g, b, 1.0])

        init_keyframe = []

        for i in range(number_of_agents):
            # Get a deep copy of quadruped body
            spec_copy = original_spec.copy()
            quadruped_body_copy = spec_copy.body("quadruped")

            # Attach it to the dummy frame of the main Spec (This is the correct way to attach bodies to specs)
            dummy_frame.attach_body(quadruped_body_copy, f"{i+1}_", "")

            # Sample a material colour
            quadruped_material = spec.material(f"{i+1}_body")
            quadruped_material.rgba = colours[i]

            # Root + quat
            init_keyframe += [0, 0, 0.55, 1.0, 0.0, 0.0, 0.0]
            # Leg joints * 4
            init_keyframe += [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]

        spec.add_key(name="init", qpos=init_keyframe)

        # Compile and print the new xml file to proof read
        model = spec.compile()
        self.model = model
        self.debug = debug

        if self.debug:
            print(spec.to_xml())

        # Let the variable "Action speed" be the speed at which our agent selects actions, we can calculate this as:
        # timestep * number of frames skipped
        # E.G.
        # action speed = 0.002 * 10 = 0.02
        # OR
        # action speed = 0.004 * 5 = 0.02
        # Notice in the two examples they have the same action speeds but have different timestep/frames skipped

        # For best results when running inference we want our agent to have the same action speed it used when it was trained

        # Change simulations iterations if you want (the policy will work best with original solver settings policy was trained on)
        # self.model.opt.ls_iterations = 50
        # self.model.opt.iterations = 50

        # For this environment we trained using a timestep of 0.004 and frames skipped was 5
        training_timestep = 0.004
        training_frame_skip = 5
        training_action_speed = training_timestep * training_frame_skip
       
        # Ideally we want all inference environments to run at a timestep of 0.002 seconds per step (to help with simulation accuracy, this becomes more important the more agents you have running at the same time)
        desired_timestep = 0.002
        desired_action_speed = training_action_speed

        # Since we have a desired timestep 0.002 and we must use the same action speed, we can calculate the ideal number of frames skipped as
        desired_frame_skip = desired_action_speed / desired_timestep

        self.model.opt.timestep = desired_timestep

        # Grab the brain neurons for backend to use
        self.brain_structure = tuple([32, 32, 32, 32])
        self.observation_dim = 27
        self.action_dim = 8
        self.frame_skip = int(desired_frame_skip)

        # Environment constants
        self.path_to_brain_neurons = "agent_brains/sprint_quadruped/model"
        self.number_of_agents = number_of_agents
        self._reset_noise_scale = 0.1
        self._finish_line_pos = model.geom("finish_line").pos

        # Sites
        agent_site_idxs = [model.site(f"{i}_quadruped").id for i in range(1,number_of_agents+1)]

        agent_qpos_idxs = [[] for _ in range(number_of_agents)]
        agent_qvel_idxs = [[] for _ in range(number_of_agents)]
        agent_qfrc_actuator_idxs = [[] for _ in range(number_of_agents)]

        curr_qpos_idx = 0
        curr_qvel_idx = 0

        # Loop through each joint in the world model
        for joint_id in range(model.njnt):
            joint_name = model.joint(joint_id).name

            # Check if joint is a root joint
            is_root = "root" in joint_name
            is_quadruped = "quadruped" in joint_name

            if is_root:
                if is_quadruped:
                    # Check which agent this root belongs to by checking the 8th char which will be its id
                    get_id = re.match(r'^(\d+)', joint_name)
                    agent_id = int(get_id.group(1)) - 1
                    agent_qpos_idxs[agent_id].extend(range(curr_qpos_idx,curr_qpos_idx + 7))
                    agent_qvel_idxs[agent_id].extend(range(curr_qvel_idx,curr_qvel_idx + 6))

                # Increment the index counters (7DOF, 3 positional, 4 rotational)
                curr_qpos_idx += 7
                curr_qvel_idx += 6

            # Include only joints starting with the agent prefix
            else:
                if is_quadruped:
                    get_id = re.match(r"^(\d+)", joint_name)
                    agent_id = int(get_id.group(1)) - 1
                    agent_qpos_idxs[agent_id].append(curr_qpos_idx)
                    agent_qvel_idxs[agent_id].append(curr_qvel_idx)
                curr_qpos_idx += 1
                curr_qvel_idx += 1

        # Loop through each DOF in the world model and store each one that belong to a agent Quadruped
        for dof_id in range(model.nv):
            for i, individual_agent_qvel_idx in enumerate(agent_qvel_idxs):
                if dof_id in individual_agent_qvel_idx:
                    agent_qfrc_actuator_idxs[i].append(dof_id)
                    break

        # Convert the lists to JAX arrays
        self._agent_qpos_idxs = jp.array(agent_qpos_idxs)
        self._agent_qvel_idxs = jp.array(agent_qvel_idxs)
        self._agent_qfrc_actuator_idxs = jp.array(agent_qfrc_actuator_idxs)
        self._agent_site_idxs = jp.array(agent_site_idxs)

        # To help quickly get x and z position of agents
        self._x_qpos_idxs = self._agent_qpos_idxs[:, 0]

        # Initial qpos for each agent
        start_poses = model.keyframe("init").qpos
        agent_qpos0_values = [start_poses[self._agent_qpos_idxs[i]] for i in range(number_of_agents)]
        self._agent_qpos0_values = jp.array(agent_qpos0_values)


    # Resets a Quadruped agent to its initial state.
    def _reset(self, data,agent_to_reset,rng: jax.Array) -> Tuple[mjx.Data, Any]:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Create initial joint position and velocities with added noise
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Reset the joint positions and velocities
        new_qpos = self._agent_qpos0_values[agent_to_reset] + jax.random.uniform(
            rng1, self._agent_qpos_idxs[agent_to_reset].shape, minval=low, maxval=hi
        )
        new_qvel = jax.random.uniform(rng2, self._agent_qvel_idxs[agent_to_reset].shape, minval=low, maxval=hi)

        rand_y = jax.random.uniform(rng2, (), minval=-4.0, maxval=4.0)
        new_qpos = new_qpos.at[1].set(rand_y)
        
        data.qpos[self._agent_qpos_idxs[agent_to_reset]] = new_qpos
        data.qvel[self._agent_qvel_idxs[agent_to_reset]]=new_qvel

        # Reset the internal actuator forces
        data.qfrc_actuator[self._agent_qfrc_actuator_idxs[agent_to_reset]] = 0.0

        # Reset the warm starts for absolute agent reset
        data.qacc_warmstart[self._agent_qvel_idxs[agent_to_reset]] = 0.0
        return data, rng

    # Resets a list of agents
    def reset(self, data: mujoco.MjData, agents_to_reset: List, rng: jax.Array, reset_all=False) -> Tuple[mujoco.MjData, jax.Array]:
        del reset_all
        if self.debug: print("resetting")

        for agent_id, reset_bool in enumerate(agents_to_reset):
            if reset_bool:
                data, rng = self._reset(data, agent_id, rng)
        return data, rng

    # This environment has no pre processing done
    def preprocess_actions(self, actions) -> jp.ndarray:
        return actions

    # Terminate if a agent Quadruped has won or if all agents have fallen
    def check_for_termination(self, data: mjx.Data) -> jp.ndarray:
        xs = data.qpos[self._agent_qpos_idxs][..., 0]

        # Get the angle of all quadruped torso's
        agent_orientations = jp.ravel(data.site_xmat[self._agent_site_idxs]).reshape(-1, 9)[:, 8]

        # Check if all agents have fallen
        cond1 = jp.all(agent_orientations < 0.3)

        # Check if a agent has won the race
        cond2 = jp.any(xs > self._finish_line_pos[0] + 5)

        reset_all_agents = jp.logical_or(cond1, cond2)

        done = jp.full((self.number_of_agents,), reset_all_agents)
        return done
  
    # The observation functions are JIT
    def _get_obs(self, data: mjx.Data, agent_id) -> jp.ndarray:
        agent_qpos = data.qpos[self._agent_qpos_idxs[agent_id]][2:]
        agent_qvel = data.qvel[self._agent_qvel_idxs[agent_id]]
        return jp.concatenate([agent_qpos, agent_qvel])

    def get_all_obs(self, data: mjx.Data) -> jp.ndarray:
        return jax.vmap(self._get_obs, in_axes=(None, 0))(data, jp.arange(self._agent_qpos_idxs.shape[0]))


