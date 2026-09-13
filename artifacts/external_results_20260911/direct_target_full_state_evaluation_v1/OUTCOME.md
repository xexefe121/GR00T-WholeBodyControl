# Final 65000 canonical outcome

The selected canonical trial failed the unchanged native joint-speed limit at control 296, substep 9, at 5.938 simulated seconds. The full 1569-control lifecycle remains incomplete. No source-motion samples or conditional 250-control hold were reached.

- Exactly one separate WSL head witness passed: one attempted/returned head call, zero BFM or physics calls.
- Exactly one canonical trial ran: 296 complete controls, one partial control, 2969 native steps, and 47 learned head calls at controls 250 through 296. Startup used the original 250 backward and 250 actor calls; learned-phase BFM calls were zero.
- Failure velocity ratio was 1.0574050513825877. Recorded range excess and accumulated clock error were zero; native warnings were zero. These are producer results, pending independent physics and semantic audits.
- Raw Python exit was 0 because the diagnostic driver saved its outcome normally. The durable wrapper correctly returned diagnostic exit 2 for the incomplete physical result. Both captured processes exited, all 5233 pinned inputs remained exact, and no retry occurred.
- Full precontrol integration, feature/history/action logs, and the strict failure capsule remain saved. The owner completion receipt verifies accounting and preservation; it does not qualify behavior.

Final head SHA256: `045f04138610a06a0171a899d002cfa4f03e30e316dc9442199c833c16e43902`.

Trace SHA256: `8eade82c93fa9c3104fcbb5807bb598005e3e19c4c6865be548906ea4adc80d2`.

Owner receipt: `evaluation_completion_verification.json`, SHA256 `3e8c5b90d0ed53d4fe627c889d01a8806cf39475cb5027f80d6dcc733d388c00`.

The parent owns independent native physics and intent replay. The reviewer owns saved feature/history semantics. No further model, optimizer, native experiment, or training change is selected by this outcome record.
