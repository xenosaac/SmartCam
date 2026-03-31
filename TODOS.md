# TODOS

## Answer History / Persistence
**What:** Append each answer + timestamp + image hash to a log file (e.g., ~/smartcam_history.txt).
**Why:** Currently answers vanish when replaced. If you miss one during a study session, it's gone. A history file lets you review all answers after the session.
**Pros:** Simple append-only file. ~5 lines of code. No new dependencies.
**Cons:** Log file grows unbounded over long use. No search/filter capability.
**Context:** Raised by outside voice during eng review (2026-03-30). The display window only shows the latest answer. For study review, you'd want to see all answers from a session.
**Depends on:** Core system working (smartcam.py).

## Prompt Engineering & Output Format
**What:** Design and test the LLM system prompt, user prompt template, and output format.
**Why:** The current prompt is a one-liner. It doesn't specify output format (plain text vs markdown vs LaTeX), solution depth (final answer only vs step-by-step), or how to handle multi-problem sheets with mixed types.
**Pros:** Better prompts → better answers. This is the core product differentiator.
**Cons:** Requires real-world testing with actual exercise sheets across subjects.
**Context:** Raised by outside voice during eng review (2026-03-30). The prompt is the most important 3 lines of code in the whole project. Iterate on it with real exercise sheets during testing.
**Depends on:** Core system working + CLIProxyAPI or Codex OAuth backend functional.
