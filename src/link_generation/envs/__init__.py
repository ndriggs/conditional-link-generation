from gymnasium.envs.registration import register

register(
    id='SignatureEnv-v0',  
    entry_point='link_generation.envs.signature_env:SignatureEnv', 
)
