from mlagents.trainers.trainer_util import load_config
from animalai_train.run_options_aai import RunOptionsAAI
from animalai_train.run_training_aai import run_training_aai
from animalai.envs.arena_config import ArenaConfig

trainer_config_path = (
    "models/ppo.yaml"
)


environment_path = "/media/pc-casa/Disk D/Camilo_Manuel/AnimalAI-Olympics/examples/env/AnimalAI"
run_id = "multiple_arena_second_training"
base_port = 5005


args = RunOptionsAAI(
    trainer_config=load_config(trainer_config_path),
    env_path=environment_path,
    run_id=run_id,
    base_port=base_port+3,
    load_model=True,
    train_model=False,
    arena_config=ArenaConfig("configurations/curriculum/8.yml")
)
run_training_aai(0, args)
