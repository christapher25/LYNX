# OWNER (only what the owner must decide or do; LEAD keeps this current)
Updated 2026-11-13, LEAD. TEAM STOPPED on your WRAP-UP AND STOP. Everything is in docs\team\HANDOVER.md in the repo.

YOUR PENDING DECISIONS (nothing runs until you restart the team):
 A. The row 21 changes list (below): approve all 341 / the 297 price-consistent only / exclusions. Recommended: all.
 B. FII flows end 2026-09-25: add the NSDL FPI ingest to data_refresh? Recommended: yes, as its own reviewed step.
 C. Paper-log backfill: your order stands (not before row 21 is merged and re-adjusted); it then needs your explicit "run the backfill".
 D. You said you would commit the job allowlist file and docs\agent-team\STAGE_QUEUE.md yourself: both are still uncommitted in the main checkout (the shell guard stops me from staging the allowlist).

## State at the stop
- R20-10 (the data_refresh announcements fix) passed all three reviews and is merged (21962b7). The data_refresh job has NOT been rerun: first thing for LEAD on restart (about 1 hour).
- data_refresh's first run (20261112_155158) failed at the announcements step; the other eight raw sources are on disk to 2026-11-11; adjusted prices, universe and features end 2026-11-09 and passed the history guard.

## A. APPROVE THE ROW 21 CHANGES LIST (nothing is applied until you answer)
The list: C:\Users\chris\PycharmProjects\LYNX_V1\logs\row21\proposed_changes.md (every line, with prices) and proposed_changes.json (sha256 544381d44a1689dc630cb79fd0e74650f26a65e27fe025849f1d4c30095109ef). Produced by the reviewed, merged script (main ead0771), run by me. It changed nothing.
What it says: 36,836 stored corporate-action rows re-parsed; 36,495 keep their factor; 341 would change, in 331 symbols:
- 337 SPLITS GAIN A FACTOR (they have none today): 164 at 0.1, 105 at 0.2, 62 at 0.5, 6 others (0.1667 x2, 0.25 x2, 0.3333, 0.4).
- 4 NON-EQUITY BONUSES LOSE THEIR FACTOR: ZEEL 2014-03-03 (preference shares), DRREDDY 2011-03-17, NTPC 2015-03-20, COROMANDEL 2012-07-13 (bonus debentures). For all four the price did NOT fall by the old factor on the ex-date.
- 0 factors change from one value to another.
Price check (close on the ex-date / previous close, against the product of all factors on that date):
- 293 splits CONSISTENT (within 25%) + the 4 bonuses consistent.
- 1 INCONSISTENT: JINDALSTEL 2008-01-21, split Rs 5 -> Re 1 (0.2): close 14,363.85 -> 2,104.25 = 0.1465, i.e. 27% below the split-implied price. 2008-01-21 was the market-crash Monday; no other action is recorded that day. My reading: a real split on a crash day. Your call.
- 43 with NO USABLE PRICE: 10 have no session before the ex-date in our data; 33 have more than 10 calendar days between the sessions around the ex-date (the symbol has no EQ-series rows there: for example CUPID 2024-04-04, PCJEWELLER 2024-12-16, TIPSMUSIC 2023-04-21). The subject text of each is an explicit face-value split. They are listed first in the .md file.
- 2 of the 341 are dated AFTER the history cutoff (JSWDULUX 2026-10-22, KMEW 2026-11-06): they do not touch history <= 2026-09-29 but belong in the same fix.
My recommendation: approve all 341. Reason: every line is an explicit face-value split in the exchange's own text (or one of the four non-equity bonuses), the factor is arithmetic on that text, and 297 of the 298 lines that can be price-checked agree. If you prefer caution: approve the 297 price-consistent lines now and hold JINDALSTEL and the 43 for a second look.
Answer in the LEAD tab: "approve all", or "approve consistent only", or name lines to exclude.
What happens when you approve: I brief the apply step (updates ONLY the parser-derived fields of the approved rows, nothing else), the guard extension that admits exactly the approved rows and their symbols' adjusted columns, then adjust_prices, the feature rebuilds, the guard, then the M3 report re-run labelled "data-corrected re-run" (M3_PREREG.md unchanged, original report kept) and the count of M1-M3 trades / candidate weeks that touched an affected symbol-date. All through the three reviews.

## Results of your 2026-11-12 decisions (done this morning)
- Circuits (A): guard extension reviewed and merged; the guard now passes with the 1,343 circuits files reported as owner-approved late fills (band columns, 2026-09-28/29 only).
- F&O and flows rebuilt; guard passed after each. No block listed any orphan file: nothing for you to delete.
- Breadth (A2): rebuilt; one guard run with breadth exempt (passed for everything else); comparison against the kept old file: HISTORY DIFFERENCES: 0. All 7,934 dates before 2026-09-30 are identical; 2026-09-30 went from missing to 42.77%; 27 dates after it were added. Your STOP condition did not trigger. Report: LYNX_V1\logs\row20_breadth_compare.md.
- Stage 10 verifier: 4 of 5 pass; flows fails only on the known publication-timing check.
- Parser fix (row 21, step 1) merged; it changed no stored data.

## One question, not urgent (answer when you like; nothing waits on it)
FII flows end on 2026-09-25: sessions from 2026-09-28 on are missing and flagged, because the NSDL monthly FPI files were not in your row-20 source list. The paper log does not read flows. Do you want data.ingest_nsdl_fpi added to the data_refresh job (and run once now)? My recommendation: yes, add it, as its own reviewed step, after R20-9.

## Your decisions of 2026-11-12, as I logged them
1. Circuits: A. Narrow reviewed guard extension: only the band columns on 2026-09-28 and 2026-09-29 where the old value was missing; a change to a non-missing old value still fails. -> task R20-7.
2. Breadth: A2 with a condition. One rebuild outside the guard, then a report of every differing date and count; if any date earlier than the old file's last date differs, STOP for you. -> task R20-8. The old file is kept at logs\history_guard\pre_rebuild\universe_breadth_before_row20.parquet (last date 2026-09-30, last value 2026-09-29).
3. Paper-log backfill: NOT yet; it waits until the split-parser fix is merged and prices are re-adjusted.
4. Split/bonus parser: new queue row 21 right after row 20 (tests first; approved-changes list; re-adjust; guard; then the M3 report re-run on corrected data with M3_PREREG.md unchanged, labelled "data-corrected re-run"; plus a count of M1-M3 trades / candidate weeks that touched an affected symbol-date).
5. Team tools: committed (bdb9ee5).
6. Allowlist commit: you run it yourself, in the main checkout, when no other commit or test run is active (the hook takes 2-3 min). Stage the job allowlist file under scripts\ and docs\agent-team\STAGE_QUEUE.md, then commit. Tell me afterwards so the devs merge main.
7. Merge-main instead of rebase: kept.
8. Loops: a tab with an empty queue stops cycling; I stop mine when I wait only on you.

## New fact for row 21 (no decision yet): the split problem is much larger than the top-200 examples
I counted the stored corporate actions (read-only): 622 rows are splits and 338 of them (54%) have NO adjustment factor, across the whole universe, all from the same two wording gaps ("Re" instead of "Rs", and "Fv Split Rs.N/- To Rs.N/" without the word "from"). Four non-equity "bonus" rows were applied as equity bonuses: ZEEL 2014-03-03 (preference shares), DRREDDY 2011-03-17, NTPC 2015-03-20, COROMANDEL 2012-07-13 (bonus debentures). So the correction will change adjusted history for roughly 300+ symbols, not a handful, and the M1-M3 inputs were affected accordingly. The process stays as you ordered: parser fix (R21-1, changes no data), then a full proposed-changes list with each factor checked against the actual ex-date price move, which you approve before anything is applied.

## What you will be asked next (so it is no surprise)
- If the breadth comparison shows any difference before 2026-09-30: I stop and show you the dates.
- Row 21 changes adjusted history for every symbol with a repaired split: I will bring you the approved-changes list (symbol, ex-date, factor) BEFORE applying it.
- One consequence of decision 3 for row 20: row 20 cannot be ticked until row 21's fix is in, because the backfill is part of row 20. I will finish everything else in row 20 first (rebuilds, verifiers, data_refresh job).
