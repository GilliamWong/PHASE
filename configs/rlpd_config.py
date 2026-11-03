from configs import sac_config


def get_config():
    config = sac_config.get_config()

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    # RLPD's settings for sparse-reward tasks (upstream used them for Adroit).
    # With the entropy backup on, the entropy bonus dominates the 0/10 success
    # reward: on Lift, Q-values reached ~240 when no return can exceed 10.
    config.backup_entropy = False
    config.hidden_dims = (256, 256, 256)

    return config
