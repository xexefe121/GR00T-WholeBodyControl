Both fixed conditions completed3000 updates, preserving ordinary68000 endpoints. Initial outputs for both conditions matched the source65000 outputs byte-for-byte across all367570 diagnostic rows. The split first layer removed the initial arithmetic drift without changing the1e-5rad threshold.

Final whole-corpus ORT64 objectives:

| Objective | Blinded | Causal | Causal change |
|---|---:|---:|---:|
| Nominal |0.000439689108559|0.000325353696826|−26.00%|
| Full-state finite feedback |0.000217593122476|0.000207214823224|−4.77%|
| Physical response |0.0000937495829662|0.0000841520354584|−10.24%|
| Fixed weighted total |0.000929201585556|0.000786392359913|−15.37%|

This measures utility of the appended causal context under the matched protocol. It does not uniquely establish hidden-state causation or separate that utility from effective input capacity. No intermediate checkpoint or rollout was used to choose these endpoints.

Counterevidence remains: causal full-state error is1.074403807734722 times the zero-response baseline0.00019286493749569113. Its odd error is0.00020287036217742077 and even error0.000004344461046468942. Better aggregate fit does not establish adequate stabilizing response or connected stability.

Both same-weight promoted exports passed the unchanged CPU64/GPU64/ORT64 preclamp gate: maximum2.438202528765032e-9rad for blinded and1.0159274116405825e-10rad for causal. GPU32 training uses the disclosed split1000+323 first layer; promoted inference uses the mathematically equivalent monolithic1323 first layer. Public input/output remainfloat32.

The pair consumed exactly6000 updates,18000 training forwards/88,116,000 rows;11496 Torch diagnostic forwards/2,940,560 rows;2874 ORT forwards/735,140 rows. There were no calibration, BFM or native calls. Historical v1 consumed a separate1437 initial forwards/367570 rows and zero updates before stopping; its failure remains preserved.

Owner verification83fcdefecb1cda0d1a24a8a018952782ed640eb7ede677fbf31686cd4c260299 confirms accounting and paired numerical completion, exact source/input/process/output hashes, raw and wrapper exit0, and absence of wrapper28060/Python5052. Per-condition18-subject owner receipts are saved. Independent saved-evidence audit is separately owned by the reviewer; simulation requires its qualification and subsequent concrete launch gates.

Causal PT:10d238198e2d7826c35eaa9bd4a13ff2e05050a7f601c1f212b585c4a90eaefd. Causal ONNX:d61915c1bf30b16431660134be6057855bbc7befeb620630a5b36dd506738f5c.

Blinded PT:d48e6b77726d8b91f65938ab31575fe9f5e115e1dafb12187814c0b579fd7e89. Blinded ONNX:8ac9551875f4de9c07f8d244e10ebe32d870477d417aef35d00420acc1c35bfc.
