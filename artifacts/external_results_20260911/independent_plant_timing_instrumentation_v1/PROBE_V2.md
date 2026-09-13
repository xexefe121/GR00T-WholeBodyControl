# Reentrant sampling correction

Version1 is preserved. Its global last-wall/process thresholds could falsely call an in-flight parent sample a regression when a nested GC callback advanced those thresholds before the parent returned its earlier captured value.

Version2 compares each sample's own wall-before/wall-after bracket, each completed span's start/end wall and CPU values, and successive completed primary-thread root spans. GC start/stop pairs use their own bracket and same-thread CPU comparisons. GC callbacks do not advance a global threshold for unrelated parent reads. Primary span ownership remains one thread; foreign-thread GC events are recorded without assigning an active primary span.

Nine additional fake tests inject GC after wall/thread/process values have been captured but before their reader returns, including foreign-thread callback intervals. They also reject real backward local brackets, paired CPU values and successive root spans. Combined31 fake tests pass. These checks verify causal pairs and observed brackets; they do not claim detection of an unobserved transient clock regression between samples.

Storage, capacities, phase schema, original-deadline separation, return-before-end-clock evidence and no real-clock/native/GC-callback execution remain unchanged. No producer source has been instrumented by this versioning step.
