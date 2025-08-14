
import mujoco
from jax import numpy as jp
from mujoco import mjx
import jax
import random
from typing import Any, Tuple, List
import re


# Inference environment for 'Sprint Humanoid'
class SprintHumanoid_Inference:
    def __init__(self, number_of_agents=1, debug=False):
        scene_path = "sports_hub/inferences/sprint_humanoid/base_model.xml"

        # Original spec to copy from
        original_spec = mujoco.MjSpec.from_file(scene_path)

        # Main Spec that we are buliding on
        spec = mujoco.MjSpec.from_file(scene_path)

        # Remove the original humanoid body form the base model
        spec.delete(spec.body("humanoid"))

        # Create a dummy frame in the main spec and use it to attach each new humanoid too
        dummy_frame = spec.worldbody.add_frame()

        # Pre compute colours for each humanoid
        colours = []

        for i in range(number_of_agents):
            r = random.random()
            g = random.random()
            b = random.random()
            colours.append([r, g, b, 1.0])

        for i in range(number_of_agents):
            # Get a deep copy of humanoid body
            spec_copy = original_spec.copy()
            humanoid_body_copy = spec_copy.body("humanoid")

            # Attach it to the dummy frame of the main Spec (This is the correct way to attach bodies to specs)
            dummy_frame.attach_body(humanoid_body_copy, f"{i+1}_", "")

            # Sample a material colour
            humanoid_material = spec.material(f"{i+1}_body")
            humanoid_material.rgba = colours[i]

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

        # For this environment we trained useing a timestep of 0.004 and frames skipped was 5
        training_timestep = 0.004
        training_frame_skip = 5
        training_action_speed = training_timestep * training_frame_skip

        # Ideally we want all inference environments to run at a timestep of 0.002 seconds per step (to help with simulation accuracy, this becomes more important the more agents you have running at the same time)
        desired_timestep = 0.002
        desired_action_speed = training_action_speed

        # Since we have a desired timestep 0.002 and we must use the same action speed, we can calculate the ideal number of frames skipped as
        desired_frame_skip = desired_action_speed / desired_timestep

        self.model.opt.timestep = desired_timestep
        self.timestep = desired_timestep

        # Grab the brain neurons for backend to use
        self.brain_structure = tuple([32, 32, 32, 32])
        self.observation_dim = 276
        self.action_dim = 17
        self.frame_skip = int(desired_frame_skip)

        # Environment constants
        self.path_to_brain_neurons = "agent_brains/sprint_humanoid/model"
        self.number_of_humanoids = number_of_agents
        self._reset_noise_scale = 0.01
        self._finish_line_pos = model.geom("finish_line").pos

        agent_qpos_idxs = [[] for _ in range(number_of_agents)]
        agent_qvel_idxs = [[] for _ in range(number_of_agents)]
        agent_qfrc_actuator_idxs = [[] for _ in range(number_of_agents)]
        agent_body_idxs = [[] for _ in range(number_of_agents)]

        curr_qpos_idx = 0
        curr_qvel_idx = 0

        # Loop through each joint in the world model
        for joint_id in range(model.njnt):
            joint_name = model.joint(joint_id).name

            # Check if joint is a root joint
            is_root = "root" in joint_name
            is_humanoid = "humanoid" in joint_name

            if is_root:
                if is_humanoid:
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
                if is_humanoid:
                    get_id = re.match(r"^(\d+)", joint_name)
                    agent_id = int(get_id.group(1)) - 1
                    agent_qpos_idxs[agent_id].append(curr_qpos_idx)
                    agent_qvel_idxs[agent_id].append(curr_qvel_idx)
                curr_qpos_idx += 1
                curr_qvel_idx += 1

        # Loop through each body in the world model
        for body_id in range(model.nbody):
            body_name = model.body(body_id).name
            is_humanoid = "humanoid" in body_name

            # Store the body indices of each agent (for Observation Space)
            if is_humanoid:
                get_id = re.match(r"^(\d+)", body_name)
                agent_id = int(get_id.group(1)) - 1
                agent_body_idxs[agent_id].append(body_id)

        # Loop through each DOF in the world model and store the ones that belong to the agent humanoid
        for dof_id in range(model.nv):
            for i, individual_agent_qvel_idx in enumerate(agent_qvel_idxs):
                if dof_id in individual_agent_qvel_idx:
                    agent_qfrc_actuator_idxs[i].append(dof_id)
                    break
        
        # Convert the lists to JAX arrays
        self._agent_qpos_idxs = jp.array(agent_qpos_idxs)
        self._agent_qvel_idxs = jp.array(agent_qvel_idxs)
        self._agent_qfrc_actuator_idxs = jp.array(agent_qfrc_actuator_idxs)
        self._agent_body_idxs = jp.array(agent_body_idxs)

        # To help quickly get x and z position of agents
        self._x_qpos_idxs = self._agent_qpos_idxs[:, 0]
        self._z_qpos_idxs = self._agent_qpos_idxs[:, 2]

        # Initial qpos for each agent
        agent_qpos0_values = [model.qpos0[self._agent_qpos_idxs[i]] for i in range(number_of_agents)]
        self._agent_qpos0_values = jp.array(agent_qpos0_values)


    # Resets a Humanoid agent to its initial state.
    def _reset(self, data,agent_to_reset,rng: jax.Array) -> Tuple[mjx.Data, Any]:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Create initial joint position and velocities with added noise
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Reset the joint positions and velocities
        new_qpos = self._agent_qpos0_values[agent_to_reset] + jax.random.uniform(rng1, self._agent_qpos_idxs[agent_to_reset].shape, minval=low,maxval=hi) 
        new_qvel = jax.random.uniform(rng2, self._agent_qvel_idxs[agent_to_reset].shape, minval=low, maxval=hi) 

        rand_y = jax.random.uniform(rng2, (), minval=-4.0, maxval=4.0)
        new_qpos = new_qpos.at[1].set(rand_y)
        
        data.qpos[self._agent_qpos_idxs[agent_to_reset]] = new_qpos
        data.qvel[self._agent_qvel_idxs[agent_to_reset]] = new_qvel

        # Reset the internal actuator forces
        data.qfrc_actuator[self._agent_qfrc_actuator_idxs[agent_to_reset]] = 0.0

        # Reset the warm starts for the fun of it
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

    # Terminate if a agent Humanoid has won or if all agents have fallen
    def check_for_termination(self, data: mjx.Data) -> jp.ndarray:
        zs = data.qpos[self._agent_qpos_idxs][..., 2]
        xs = data.qpos[self._agent_qpos_idxs][..., 0]

        # Check if all agents have fallen
        cond1 = jp.all(zs < 0.3)

        # Check if any agent has finished the race
        cond2 = jp.any(xs > self._finish_line_pos[0] + 5)

        reset_all_agents = jp.logical_or(cond1, cond2)

        done = jp.full((self.number_of_humanoids,), reset_all_agents)
        return done
  
    # The observation functions are JIT
    def _get_obs(self, data: mjx.Data, agent_id) -> jp.ndarray:
        agent_qpos = data.qpos[self._agent_qpos_idxs[agent_id]][2:]
        agent_qvel = data.qvel[self._agent_qvel_idxs[agent_id]]
        agent_cinert = data.cinert[self._agent_body_idxs[agent_id]].ravel()
        agent_cvel = data.cvel[self._agent_body_idxs[agent_id]].ravel()
        agent_qfrc_actuator = data.qfrc_actuator[self._agent_qfrc_actuator_idxs[agent_id]]

        return jp.concatenate(
            [agent_qpos, agent_qvel, agent_cinert, agent_cvel, agent_qfrc_actuator]
        )

    def get_all_obs(self, data: mjx.Data) -> jp.ndarray:
        return jax.vmap(self._get_obs, in_axes=(None, 0))(data, jp.arange(self._agent_qpos_idxs.shape[0]))


