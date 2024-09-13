from ray.rllib.algorithms.ppo import PPOConfig

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("LinkBuilderEnv-v0")
    .env_runners(num_env_runners=2)
    .framework("torch")
    .evaluation(evaluation_num_env_runners=1)
)

algo = config.build()  # 2. build the algorithm,

for _ in range(5):
    print(algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.