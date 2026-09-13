Initial loader assumed both historical report families stored trace_sha256 at
top level. The earlier original29 reports instead bind the exact trace path in
their inputs manifest. Setup exited before fit/output-directory creation or
any network inference. The corrected loader verifies that existing SHA pin;
it does not bypass validation or change recordings, labels or fit settings.
