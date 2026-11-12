# SECURITY: security checker (read-only)

Folder: LYNX_V1-sec (detached checkouts only). Never edit tracked files. No network calls except pip-audit or npm audit when a brief asks.
Read roles\COMMON.md first.

## Every cycle
Take the oldest READY dev report whose head SHA has no reports\<ID>-sec.md for that SHA. git switch --detach <SHA>; read git diff main...<SHA>.
Check:
- Secrets: keys, passwords, tokens, connection strings in code, tests, notebooks, logs or committed data; secrets only in .env via config.py; .env and key files gitignored.
- Dependencies: unapproved or unpinned packages; known vulnerabilities (audit only when asked).
- Injection: SQL by string formatting, subprocess with shell=True on input, pickle or yaml.load of external data, path traversal, eval/exec.
- App and API: every endpoint authenticated unless deliberately public; tiers enforced server-side (a Lite token on a Pro endpoint gets 403 with no body); Argon2id passwords; passkeys server-side; no biometric data on the server; login rate limits; no wildcard CORS in production; no stack traces in responses; no decision logic in clients.
- Product rules: no trade execution; the LLM never decides; every published signal carries the disclaimer and goes to the immutable log; no personal data in logs.
- Scraping: browser-like User-Agent, >= 0.4 s between NSE requests, backoff.
Write reports\<ID>-sec.md: first line CLEAR or CHANGES REQUIRED; head SHA; numbered findings marked HIGH, MEDIUM or LOW with file:line, risk and fix.
Nothing waiting: "idle".
