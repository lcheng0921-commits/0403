from algo.mha_mb_ppo.run_mbppo import train
from experiment.mb_ppo.maps import ClusteredMap500


if __name__ == '__main__':
    train_kwargs = {
        'map': ClusteredMap500,
        'baseline': 'mbppo',
        'single_head': False,
        'rsma_mode': 'rsma',
        'episodes': 1000,
        'eval_interval': 20,
        'save_freq': 100,
    }
    train(train_kwargs=train_kwargs)
