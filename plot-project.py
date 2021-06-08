import pandas as pd
import matplotlib.pyplot as plt

name = "multiple_arena_second"

df = pd.read_csv(f"summaries/{name}_training_AnimalAI.csv")
cummultatuve_reward = pd.to_numeric(df["Environment/Cumulative Reward"].replace("None", "0"))

plt.figure()
plt.plot(cummultatuve_reward)
plt.xlabel("Steps (x10e3)")
plt.ylabel("Cumulative Reward")
plt.grid()
plt.show()
