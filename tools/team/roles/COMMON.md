# Rules for every role on the LYNX tab team

Read this once when your session starts, and again after a /compact or when unsure.

## Where things are
- Team folder: C:\Users\chris\PycharmProjects\LYNX-team (BOARD.md, STATE.md, OWNER.md, DECISIONS-OWNER.md, tasks\, reports\, out\, roles\, bin\).
- Main checkout C:\Users\chris\PycharmProjects\LYNX_V1 belongs to LEAD only. Every other role works in its own worktree: LYNX_V1-devA, LYNX_V1-devB, LYNX_V1-test, LYNX_V1-verify, LYNX_V1-sec.
- Sources of truth, in order: CLAUDE.md; docs/LYNX-v6-Final-Architecture-and-Setup-Documentation.txt with its dated amendments; docs/vanta/VANTA_SPEC.md, INTERFACES.md and the M*_PREREG.md files; docs/agent-team/STANDING_ORDERS.md (authority order, NEEDS USER list, job rules); then existing code and tests.

## Talking to each other
- Only LEAD edits BOARD.md, STATE.md, OWNER.md and tasks\*.md. The owner talks only to LEAD.
- Everyone else writes only in reports\: <ID>-dev.md, <ID>-test.md, <ID>-sec.md, <ID>-verify.md, and NOTE-<ROLE>-<yyyymmdd-hhmm>.md for a question, a blocker or a gap.
- Every report starts with: Task: / Role: / Branch: / Head SHA: / Result: READY | APPROVED | CLEAR | CHANGES REQUIRED | BLOCKED | PLANNING / Date:
- A review counts only for the head SHA it names. A new commit needs new test, security and verify reports.

## Data, jobs and memory (16 GB laptop)
- data\ in every worktree is a junction to the main checkout's data\. It is SHARED and REAL. Outside the main checkout never run anything that writes under data\ or logs\jobs\, never start scripts\run_job.py jobs, and never run a full-universe or full-history pipeline. Use test fixtures and small read-only ranges.
- Long jobs run only in the main checkout, only by LEAD, only through scripts\run_job.py.
- Before anything heavy (the full invariant suite, a model fit), check free RAM: (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB must be >= 3 (GB). If not, do reading or planning and retry next cycle. Never run two full invariant suites at once: only TESTER runs the full suite; others run targeted tests.
- Use the main venv: python is C:\Users\chris\PycharmProjects\LYNX_V1\.venv\Scripts\python.exe (already active in your tab).

## Git (local only)
- Nobody pushes. Ever. Branches are shared between worktrees automatically.
- Devs commit on their own task branch in their worktree. Reviewers use detached checkouts of the SHA under review. Only LEAD merges into main, in the main checkout.
- Never --no-verify, never change core.hooksPath, never reset --hard, clean, or rebase main.

## Hard rules
CLAUDE.md applies to everyone, always: signal only, no trade execution, the LLM never decides, no Viterbi or smoothed states, no zero-fill, secrets only via config.py and .env, pre-registered parameters never changed after results are seen. No new dependency unless the brief allows it.

## Questions
Write a NOTE and carry on with something else. Never guess a result-changing rule (labels, costs, exclusions, thresholds, gate parameters). LEAD answers from the sources or puts it in OWNER.md.

## Cycles
You run under /loop. Each cycle:
1. Read BOARD.md and the reports and NOTEs that concern you.
2. Do the next piece of your work.
3. End with at most three lines of status. Nothing to do: end with the single word "idle".
Do not re-read the docs every cycle; read them when you start a new task. If your context is getting long, write a handoff into your report and a NOTE asking LEAD to have your tab restarted.
