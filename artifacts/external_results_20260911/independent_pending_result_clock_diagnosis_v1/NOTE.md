# Job 1211 was already expired when the plant created it

The worker correctly rejected job1211 before computing a reply. Its original deadline was147845420673ns. The plant created it at147846835414ns, **1.414741ms late**, and publication returned at147847049731ns,1.629058ms late. The worker took the job at147848090606ns and recorded `EXPIRED_BEFORE_REPLY` at147848820181ns. No reply or result-publication attempt occurred for that job.

The preceding physics tick12099 started only37,441ns after its nominal start but took **23,076,358ns** through captured completion. It finished21,113,799ns past its2ms end deadline; outer accounting returned another9,900ns later. The next control-boundary job was consequently created after its unchanged activation deadline. This establishes the saved ordering and delay. The data does not identify GC, native compute or OS preemption as the cause of the23ms body interval.

Earlier timing failure remains separate: first missed step index4800 woke26,475ns late and spent1,998,884ns in its body, ending25,359ns late. The run had40 captured-step misses and the same40 outer-cycle misses. Maximum recorded debt was10 steps. Median body517,092ns, p95 about1,015,795ns, p99 about1,385,610ns; maximum was the23.076ms tick above.

Plant job36 recovered one BUSY publication using the same bytes and original deadline. **All1,210 worker-result publication attempts returned PUBLISHED; worker-result BUSY count was zero.** The new worker retry path was therefore not exercised by this trial. The worker decoded1,211 jobs, computed1,210 replies and recorded1,211 terminal events. Pending/last-result evidence retains the expired job with zero attempts.

The missed command latched the prior command through controls1211..1266,56 held controls. Strict native joint-bound verification failed at step12663; all12663 attempted steps returned and were captured, with12662 verified. Full18190-step scope remains incomplete. Allfour model serializations matched; no watchdog timeout occurred. Known raw/diagnostic/wrapper exit2 and complete process-absence accounting were preserved.

This diagnosis reads bound saved evidence only and rehashes its inputs. The separate full saved audit was pending when `report.json` was written. Neither saved diagnosis nor future evidence-integrity acceptance makes this timing, command or physical result pass. No new clock, physics, model, optimizer or worker execution occurred. No next timed run is selected here.
