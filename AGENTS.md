# Notes for coding agents

- Start with `README.md` (usage) and `docs/method.md` (what each stage computes). Keep changes small and in the
  style of the surrounding code: plain functions, numpy, few dependencies, no framework in the browser UI.
- If a git-ignored `private/AGENTS.md` exists, read and follow it too; never copy anything from `private/` into
  tracked files.
- Test with `python -m pytest` (about 30 s). `tests/test_end_to_end.py` runs the whole pipeline on a synthetic case;
  a change to detection, physics or matching must keep it passing. Synthetic tests only prove internal consistency:
  never present them as accuracy on real recordings.
- Report accuracy from the bootstrap and leave-one-out spread and from validation against an independent
  reference, never from optimiser precision.
- Blind runs: `match/result.json` is stamped (`result.sha256`) before any reference is used. Never edit it by hand;
  re-run `adoppler match --force`. Do not look up or use a reference position before the prediction exists.
- Privacy: never commit or publish real recordings, their case folders or estimated positions of other people's
  microphones. `cases/` and `data/` are git-ignored; `adoppler export` refuses real cases unless `--allow-real`
  is passed with the owner's permission.
- Downloads (flight data, elevation) are untrusted input: keep them inside the case folder and parse, never
  execute, them.
- The local server (`adoppler/server.py`) is a security boundary: keep the Host/Origin checks, the `X-Adoppler`
  header on writes, the case-ID pattern and the CSP (no inline scripts, no `unsafe-eval`) when adding endpoints,
  and add a test in `tests/test_server.py`.
- `site/` is the Vercel demo. `tools/build_site.py` regenerates `site/demo/` and the demo numbers in
  `site/index.html`; edit the text there, not the generated table rows.
