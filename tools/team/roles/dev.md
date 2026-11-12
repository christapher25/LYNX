# DEV-A, DEV-B: developers

Folder: LYNX_V1-devA or LYNX_V1-devB (your role name says which). Read roles\COMMON.md and CLAUDE.md first.
Your work is the rows of BOARD.md with your role in the Lane column.

## Every cycle, in this order
1. Changes first. If one of your tasks is in changes: switch to its branch, do the required "Round N fixes" in tasks\<ID>.md, run the targeted tests, commit, and update reports\<ID>-dev.md with a Round N section, the new head SHA and Result: READY.
2. Otherwise start your next assigned task whose dependencies are merged:
   - git fetch is not needed (branches are shared); make sure your tree is clean; git switch -c <branch from the brief> main.
   - Read the brief and the source sections it names. Write reports\<ID>-dev.md with Result: PLANNING and a plan of at most 15 lines, then start.
   - Tests first (prove each fails without your code), then code. Small commits. Targeted tests only: python -m pytest <your test files> -q, plus python -m pytest tests/invariants -q -x only if free RAM >= 3 GB and TESTER is not running a full suite.
   - Mutation proofs from the brief: apply, run, see the named test fail, restore with git checkout -- <file>, confirm git status is clean.
3. When done: git rebase main (your branch only), rerun the targeted tests, and write reports\<ID>-dev.md: Result: READY, branch, head SHA, the commands you ran with results, tests added, mutation proofs and the test that caught each, any long job LEAD must run (module, arguments, runtime, passing check), and every point where you differed from the brief.
4. While waiting for review you may start your next assigned task, only from a clean tree.
5. Nothing to do: "idle".

## Never
Edit files outside the brief's Allowed paths, shared files (config.py, CLAUDE.md, scripts\jobs.json, requirements, .claude\), or the team folder except your own reports and out\<ID>\. Merge, push, add a dependency the brief does not allow, weaken or skip a test, run long jobs, or write under data\.
