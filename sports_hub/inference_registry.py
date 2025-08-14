# env_registry.py

from sports_hub.inferences.sprint_humanoid.env import SprintHumanoid_Inference
from sports_hub.inferences.sprint_quadruped.env import SprintQuadruped_Inference
from sports_hub.inferences.longjump_humanoid.env import LongJumpHumanoid_Inference
from sports_hub.inferences.longjump_quadruped.env import LongJumpQuadruped_Inference
from sports_hub.inferences.gaps_humanoid.env import GapsHumanoid_Inference
from sports_hub.inferences.hurdles_humanoid.env import HurdlesHumanoid_Inference
from sports_hub.inferences.fetch_humanoid.env import FetchHumanoid_Inference
from sports_hub.inferences.fetch_quadruped.env import FetchQuadruped_Inference

ENV_REGISTRY = {
    "sprint_humanoid": SprintHumanoid_Inference,
    "sprint_quadruped": SprintQuadruped_Inference,
    "longjump_humanoid": LongJumpHumanoid_Inference,
    "longjump_quadruped": LongJumpQuadruped_Inference,
    "gaps_humanoid": GapsHumanoid_Inference,
    "hurdle_humanoid": HurdlesHumanoid_Inference,
    "fetch_humanoid": FetchHumanoid_Inference,
    "fetch_quadruped": FetchQuadruped_Inference,
}
