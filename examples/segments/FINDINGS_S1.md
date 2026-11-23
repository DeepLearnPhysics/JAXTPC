# Systematic Hyperparameter Study — Sobolev s=1.0

Base config from s=1.5 sweeps on mpvmpr_20.h5:
```
lr=1.0, decay=0.999, lr_e_mult=0.003, warmup=100, noise_lr=0.3, l1=0.0
reloc_every=25, max_reloc=100, n_seg=10000, steps=1500
recomb=modified_box
```

Now testing with s=1.0 on out.h5 (0.3mm segments, forced dx=0.3mm for truth+optimizer match).
Starting with event 2 (94,886 segments, 5717.3 MeV) as it's the smallest.

## lr_emult: lr_e_mult sweep (s=1.0)
File: out.h5, event=2, dx=0.3mm, s=1.0

| lr_e_mult | Loss | dE_ratio | Dead | Time |
|---|---|---|---|---|
| 0.001 | 0.115041 | 2.104 | 0 | 388s |
| 0.003 | 0.049505 | 3.595 | 0 | 386s |
| 0.005 | 0.029731 | 4.629 | 0 | 396s |
| 0.01 | 0.015021 | 6.251 | 0 | 400s |


## noise: Noise sweep (s=1.0)
File: out.h5, event=2, dx=0.3mm, s=1.0

| noise_lr | Loss | dE_ratio | Dead | Time |
|---|---|---|---|---|
| 0.0 | 0.053193 | 3.593 | 0 | 366s |
| 0.1 | 0.052763 | 3.594 | 0 | 370s |
| 0.3 | 0.049715 | 3.595 | 0 | 377s |
| 0.5 | 0.046956 | 3.594 | 0 | 383s |
| 1.0 | 0.045439 | 3.591 | 0 | 641s |


## lr_e_mult (s=1.0, out.h5): lr_e_mult sweep at 50k segments
out.h5 ev2, 50k segs, s=1.0, dx=0.3mm, 1000 steps, track jitter 50mm

| Config | Loss | Q_ratio | Dead |
|---|---|---|---|
| lr_e_mult=0.001 | 0.000331 | 1.001 | 4940 |
| lr_e_mult=0.003 | 0.000150 | 1.000 | 11943 |
| lr_e_mult=0.005 | 0.000117 | 1.000 | 15143 |
| lr_e_mult=0.01 | 0.000095 | 1.000 | 18979 |


## lr_decay (s=1.0, out.h5): lr × decay sweep (lr_e_mult=0.01)
out.h5 ev2, 50k segs, s=1.0, dx=0.3mm, 1000 steps, track jitter 50mm

| Config | Loss | Q_ratio | Dead |
|---|---|---|---|
| lr=0.5, decay_rate=0.9995 | 0.000184 | 1.000 | 18756 |
| lr=0.7, decay_rate=0.999 | 0.000151 | 1.000 | 18551 |
| lr=1.0, decay_rate=0.999 | 0.000095 | 1.000 | 18964 |
| lr=1.5, decay_rate=0.998 | 0.000090 | 1.000 | 18098 |
| lr=2.0, decay_rate=0.997 | 0.000097 | 1.000 | 17150 |


## noise (s=1.0, out.h5): noise sweep (lr_e_mult=0.01, {'lr': 1.5, 'decay_rate': 0.998})
out.h5 ev2, 50k segs, s=1.0, dx=0.3mm, 1000 steps, track jitter 50mm

| Config | Loss | Q_ratio | Dead |
|---|---|---|---|
| noise_lr=0.0 | 0.000096 | 1.000 | 17558 |
| noise_lr=0.1 | 0.000094 | 1.000 | 17751 |
| noise_lr=0.3 | 0.000090 | 1.000 | 18122 |
| noise_lr=0.5 | 0.000097 | 1.000 | 18633 |
| noise_lr=1.0 | 0.000139 | 1.000 | 20822 |


## relocation (s=1.0, out.h5): relocation sweep (all best so far)
out.h5 ev2, 50k segs, s=1.0, dx=0.3mm, 1000 steps, track jitter 50mm

| Config | Loss | Q_ratio | Dead |
|---|---|---|---|
| reloc_every=25, max_reloc=50 | 0.000099 | 1.000 | 18565 |
| reloc_every=25, max_reloc=100 | 0.000090 | 1.000 | 18096 |
| reloc_every=25, max_reloc=200 | 0.000077 | 1.000 | 16788 |
| reloc_every=25, max_reloc=500 | 0.000058 | 1.000 | 11164 |
| reloc_every=10, max_reloc=100 | 0.000074 | 1.000 | 16073 |
| reloc_every=10, max_reloc=200 | 0.000060 | 1.000 | 11290 |


## relocation_big: Push max_reloc higher
out.h5 ev2, 50k, s=1.0, lr=1.0, d=0.999, e_mult=0.01, noise=0.3, 1000 steps

| Config | Loss | Q_ratio | Dead | Relocs |
|---|---|---|---|---|
| every=25, max=500 | 0.000063 | 1.000 | 12549 | 18000 |
| every=25, max=750 | 0.000060 | 0.999 | 6495 | 27000 |
| every=25, max=1000 | 0.000056 | 1.000 | 368 | 36000 |

