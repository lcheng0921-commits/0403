# UAV-RSMA MB-PPO (Paper-Draft Focused)

This repository is now trimmed to the MB-PPO pipeline used for the paper draft workflow.

Scope kept in code:
- MB-PPO training/evaluation pipeline
- RSMA-aware environment and physical mapping
- baseline comparisons: `mbppo`, `ppo`, `circular`, `hover`, `sdma`, `noma`
- plotting and sweep scripts for draft figures

Legacy DRQN and old experiment folders were removed.

## Environment Setup

Create the conda environment:
```
conda env create -f environment.yaml
conda activate uav_rsma
```

Install GPU PyTorch (Windows example):
```
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

Quick check:
```
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Train MB-PPO

Default MB-PPO run:
```
python -m algo.mb_ppo.run_mbppo --baseline mbppo --episodes 1000 --eval-interval 20 --save-freq 100
```

With custom QoS and max power:
```
python -m algo.mb_ppo.run_mbppo --baseline mbppo --episodes 400 --qos-threshold 0.6 --tx-power-max-dbm 28
```

With physical-mapping controls:
```
python -m algo.mb_ppo.run_mbppo --baseline mbppo --episodes 400 --phy-mapping-blend 0.8 --precoding-gain-scale 1.2 --interference-scale 1.1
```

## Run Baselines

PPO baseline:
```
python -m algo.mb_ppo.run_mbppo --baseline ppo --episodes 1000
```

Circular trajectory:
```
python -m algo.mb_ppo.run_mbppo --baseline circular --episodes 1000
```

Hovering trajectory:
```
python -m algo.mb_ppo.run_mbppo --baseline hover --episodes 1000
```

SDMA mode:
```
python -m algo.mb_ppo.run_mbppo --baseline sdma --episodes 1000
```

NOMA mode:
```
python -m algo.mb_ppo.run_mbppo --baseline noma --episodes 1000
```

## Draw Figures

Generate all target figures from `mb_ppo_data`:
```
python .\experiment\mb_ppo\draw.py
```

Use selected experiment ids:
```
python .\experiment\mb_ppo\draw.py --exp-ids 1 2 3
```

## Sweep Experiments

Run compact sweeps:
```
python .\experiment\mb_ppo\sweep.py --mode both
```

Examples:
```
python .\experiment\mb_ppo\sweep.py --mode qos --baselines mbppo ppo sdma --qos-values 0.3 0.5 0.7 --episodes 200
python .\experiment\mb_ppo\sweep.py --mode power --baselines mbppo ppo sdma --power-values 24 27 30 33 --episodes 200
```

## Output Structure

Training outputs are stored in:
- `mb_ppo_data/expX/config.json`
- `mb_ppo_data/expX/checkpoints/`
- `mb_ppo_data/expX/logs/`
- `mb_ppo_data/expX/vars/`

Draft figures are saved under:
- `experiment/mb_ppo/pics/`
