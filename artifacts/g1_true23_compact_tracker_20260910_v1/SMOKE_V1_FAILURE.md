# Setup failure, no training

Smoke89575 stopped before actor construction or optimizer steps. MJLab's
model config supplies `cnn_cfg=None`; the initial actor signature rejected it.
The native23 environment, action and source parity checks had completed.

Preserve v1 driver/actor and `smoke_v1/outcome.json` unchanged. Version2 adds
an explicit None-only config shim; same network, rewards, data and physics.
Version2 also saves measured root/feet fields for independent feature audits.
