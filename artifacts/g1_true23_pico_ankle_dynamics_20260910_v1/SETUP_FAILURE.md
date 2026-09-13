Initial capture96479 EXIT1 after input validation and reference reset, before
the first replay physics step: nine accumulator variables received ten lists.
Original capture.py and capture_v1/request.json remain unchanged. Version2
changes only both accumulator list counts from10 to9 and uses a separate output
directory. The ten-substep physics/guard horizon is unchanged. No source, model,
controller, reward or limit change. Seven force-observer tests pass.
