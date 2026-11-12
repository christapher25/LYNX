# STATE (LEAD keeps this current)
2026-11-13, LEAD. TEAM STOPPED on the owner's WRAP-UP AND STOP. Twelve tasks merged (R20-1..R20-10, R21-1, R21-2); no unreviewed work on any branch. No job running; all loops stopped. Full handover: docs\team\HANDOVER.md.
Row 20: data rebuilt to 2026-11-09 with the history guard passing (1,475 allowed differences, all owner-approved kinds). Eight raw sources are on disk to 2026-11-11; the data_refresh job has not yet completed a run (first run failed at announcements; fix R20-10 is merged; rerun not done). Paper-log backfill not run (owner: after row 21).
Row 21: parser fixed; proposed-changes list of 341 lines awaits the owner's approval; nothing applied.

Next three steps on restart:
1. LEAD: rerun data_refresh (about 1 hour, RAM gate 2 GB), check the guard.
2. Owner: approve the row 21 list. LEAD: briefs R21-3.. .
3. After row 21's re-adjustment and the M3 data-corrected re-run: paper-log backfill on the owner's explicit go.
