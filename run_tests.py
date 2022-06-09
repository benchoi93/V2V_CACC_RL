import os
from re import A
import subprocess

actor_list = os.listdir("models_server/run19/logs")
print(actor_list)
for actor in actor_list:
    print(actor)
    process = subprocess.Popen(
        ["python", "test.py",
         "--num_processes", "4",
         "--render",
         "--render_interval", "1",
         "--td",
         "--bu",
         "--shared-reward",
         "--exploration_noise", "0.03",
         "--ss-reward-coef", "10",
         "--model_path",
         f"models_server/run19/logs/{actor}",
         "--max_episode", "1"
         ]
    )

    process.wait()
    print(f"{actor} done")
