from gymnasium.envs.registration import register

register(
    id='SignatureEnv-v0',  
    entry_point='link_generation.envs.signature_env:SignatureEnv', 
)

register(
    id='SigDetEnv-v0',  
    entry_point='link_generation.envs.sig_det_env:SigDetEnv', 
)