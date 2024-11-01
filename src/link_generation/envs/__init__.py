from gymnasium.envs.registration import register

register(
    id='LinkBuilderEnv-v0',  
    entry_point='envs.signature_environment:LinkBuilderEnv', 
)
