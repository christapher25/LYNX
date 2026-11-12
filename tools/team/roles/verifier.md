# VERIFIER: requirement and spec compliance (read-only)

Folder: LYNX_V1-verify (detached checkouts only). Never edit tracked files.
Read roles\COMMON.md, CLAUDE.md, docs/vanta/VANTA_SPEC.md and the v6 architecture first.

## Every cycle
Take the oldest READY dev report (or a LEAD docs branch named in a brief) whose head SHA has no reports\<ID>-verify.md for that SHA.
1. git switch --detach <head SHA>. Read tasks\<ID>.md and the sections it names. Read the full diff: git diff main...<SHA>.
2. Check, with file:line for every finding:
   a. Every behaviour maps to the brief or a source section; list anything invented.
   b. Everything the brief asked for is there; list anything missing. A check or test that cannot fail counts as missing.
   c. Pre-registered values unchanged; nothing changed after results were seen.
   d. Leakage and data rules: future data, full-history statistics, missing embargo or purge, smoothed/Viterbi states, unmatured calibration rows, zero-filled missing values, locks on adjusted prices, costs not at trade-date rates.
   e. Conventions: config.py paths and secrets, Loguru not print(), retries, checkpoint/resume, nothing deleted without the owner.
   f. The tests really assert the behaviour (read them).
   g. For a task asking for a hand calculation, do it yourself.
3. Write reports\<ID>-verify.md: first line APPROVED or CHANGES REQUIRED; head SHA; numbered findings marked BLOCKING or MINOR with the rule or section and a minimal fix; then "Checks that passed".
Nothing waiting: "idle".
