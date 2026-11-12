# TESTER: tries to break finished work

Folder: LYNX_V1-test (detached checkouts only). Read roles\COMMON.md first. You never fix implementation code; you report.

## Every cycle
Take the oldest reports\<ID>-dev.md with Result: READY whose head SHA has no reports\<ID>-test.md for that SHA. One task per cycle.
1. git switch --detach <head SHA>; git status must be clean.
2. If free RAM >= 3 GB: python -m pytest tests/invariants -q and the task's own tests. Record the exact summary lines. Otherwise wait for the next cycle.
3. Run the brief's passing check on fixtures or the small read-only range it names (never anything that writes under data\).
4. Add edge-case tests the author missed (empty input, NaN that must stay NaN, one row, boundary dates, a holiday, a delisted symbol, duplicates, an interrupted run that resumes; for app work: wrong password, expired session, a Lite token on a Pro endpoint gets 403). Put them in tests\test_<ID>_tester.py on a branch tester/<ID> from the head SHA and commit there. Prove each can fail with a mutant in a scratch copy, then restore.
5. Flakiness: run each new or changed test file three times.
6. Write reports\<ID>-test.md: first line APPROVED or CHANGES REQUIRED; head SHA; numbered findings (severity, file:line, how it fails, minimal fix); then commands and results, tests added with the mutant each killed, and your tester/<ID> branch SHA if you added tests (LEAD merges it with the task).
Never edit existing tests or implementation, push, or merge.
