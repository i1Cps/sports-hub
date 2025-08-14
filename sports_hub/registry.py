from brax import envs

# ----------------------------------------
# Explicit imports of environments and evals
# ----------------------------------------

# Sprint environments
from sports_hub.environments.sprint_humanoid.env import HumanoidSprint, HumanoidSprint_Eval
from sports_hub.environments.sprint_quadruped.env import QuadrupedSprint, QuadrupedSprint_Eval

# Long jump environments
from sports_hub.environments.longjump_humanoid.env import HumanoidLongJump, HumanoidLongJump_Eval 
from sports_hub.environments.longjump_quadruped.env import QuadrupedLongJump, QuadrupedLongJump_Eval 

# Gaps environments
from sports_hub.environments.gaps_humanoid.env import HumanoidGaps, HumanoidGaps_Eval

# Hurdle environments
from sports_hub.environments.hurdles_humanoid.env import  HumanoidHurdles, HumanoidHurdles_Eval 

# Fetch environments
from sports_hub.environments.fetch_quadruped.env import QuadrupedFetch, QuadrupedFetch_Eval
from sports_hub.environments.fetch_humanoid.env import HumanoidFetch, HumanoidFetch_Eval

def register_all():

    # Sprint
    envs.register_environment("sprint_humanoid", HumanoidSprint)
    envs.register_environment("sprint_humanoid_eval", HumanoidSprint_Eval)
    envs.register_environment("sprint_quadruped", QuadrupedSprint)
    envs.register_environment("sprint_quadruped_eval", QuadrupedSprint_Eval)

    # Long jump
    envs.register_environment("longjump_humanoid", HumanoidLongJump)
    envs.register_environment("longjump_humanoid_eval", HumanoidLongJump_Eval)
    envs.register_environment("longjump_quadruped", QuadrupedLongJump)
    envs.register_environment("longjump_quadruped_eval", QuadrupedLongJump_Eval)

    # Gaps
    envs.register_environment("gaps_humanoid", HumanoidGaps)
    envs.register_environment("gaps_humanoid_eval", HumanoidGaps_Eval)

    # Hurdle
    envs.register_environment("hurdle_humanoid", HumanoidHurdles)
    envs.register_environment("hurdle_humanoid_eval", HumanoidHurdles_Eval)
    
    # Fetch
    envs.register_environment("fetch_quadruped", QuadrupedFetch)
    envs.register_environment("fetch_quadruped_eval", QuadrupedFetch_Eval)
    envs.register_environment("fetch_humanoid", HumanoidFetch)
    envs.register_environment("fetch_humanoid_eval", HumanoidFetch_Eval)

# Auto-register on import
register_all()
