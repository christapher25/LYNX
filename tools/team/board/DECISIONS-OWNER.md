# Owner decisions for LEAD (2026-11-10). Log each one as the owner's in logs\agent_team_decisions.md.

Context: row 20 (data catch-up) stopped at a history-guard failure. The handover note is at the end of logs\agent_team_decisions.md and row 20 in docs\agent-team\STAGE_QUEUE.md. The corporate-actions refresh mode, its tests and the audit script are UNCOMMITTED in the main checkout.

1. Guard and ingester: do both (a) and (b). (a) is narrow: in the corporate-actions dataset only, the guard treats a dtype change on a column that is EMPTY in every row dated <= 2026-09-29 as no change. Every non-empty value, the row set and the ex-dates stay strictly compared. Adjusted prices and features stay strict. (b) the refresh keeps each file's stored schema. Tests first for both.
2. Add the AASTHA bonus (ex-date 2026-09-28, factor 0.5) and the 112 pre-cutoff dividends. Log it as a DATA CORRECTION of history dated <= 2026-09-29, approved by the owner, with AASTHA added to the guard's expected-change list. In the decision log, note that M1-M3 ran without this bonus and state whether AASTHA was ever a candidate (liquidity top 200) in any M1-M3 judged week.
3. Re-run adjust_prices for the affected symbols, then history_guard_verify. Expected differences: adjusted-price columns for BLSE, BUILDPRO, CONCORDBIO, MOLDTECH, MOLDTKPAC and AASTHA only. Anything else: STOP and put it in OWNER.md.
4. The 21,321 moves above 20% without a matching action: change nothing. Report (i) the same count from the pre-refresh state, (ii) how many fall after 2026-09-29, (iii) how many fall on symbol-days in the liquidity top 200 in any M1-M3 judged week. If (iii) is non-zero, list the 20 largest and STOP for the owner before the feature rebuild.
5. Docker/Postgres stays down; the indicators table stays out of scope.
Then: universe and feature rebuilds with --list-orphans-only, history_guard_verify, the verifiers, the data_refresh job, and the paper-log backfill (wrapper, --allow-late, rows late=True). STOP for the owner at the end or at any guard failure.
RAM gates: 2 GB for ingestion and downloads, 3 GB for anything that trains, scores or runs the full invariant suite.
