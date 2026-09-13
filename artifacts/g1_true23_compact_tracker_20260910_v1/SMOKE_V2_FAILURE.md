# Second setup failure, no training

Actor construction now passed; stock RSL critic rejected the same MJLab-only
`cnn_cfg=None` field. No optimizer updates or rollout steps occurred. Version3
normalizes both model configs before construction, rejecting non-None CNNs.
Old sources, manifests and outcomes remain unchanged.
