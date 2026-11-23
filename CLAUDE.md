# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

_Add your project-specific conventions here_

## 📂 Knowledge Bundle / OKF
Architecture, contracts and procedures for this repository are documented in the OKF bundle at `okf/` (start at `okf/index.md`). Run OKF tools from the repo root. If nested bundles exist, use the closest parent `okf/` of the file you are working on. At session start, run `node C:/Users/kvlpv/.claude/skills/okf-lint/scripts/lint.js --drift`; if it reports drift, run `node C:/Users/kvlpv/.claude/skills/okf-maintain/scripts/maintain.js`.

## ⚡ Pre-Work Grounding Gate (MANDATORY)
Before answering questions about the system, planning, or editing:
1. **Search:** `node C:/Users/kvlpv/.claude/skills/okf-query/scripts/query.js --search <keywords>` (or read `okf/index.md`).
2. **Per-file context:** `node C:/Users/kvlpv/.claude/skills/okf-query/scripts/query.js --file <path[,path2,...]>` before analyzing or editing files (batch multiple paths in one comma-separated call); if nothing matches, search by module name.
3. **Cite** the concept(s) you used, or state "OKF Check: no matching concept for <topic>".
4. **Model-Tier Routing:** use the model's **High** setting for bootstrapping, architecture discovery and planning; switch to **Medium** once the plan is approved, for implementation and upkeep.

## ⚡ Pre-Completion Verification Gate (MANDATORY)
Before reporting work complete after changing code:
1. **Semantic Concept & Source Binding Upkeep:** update affected concepts, or create one, when you change a schema, component, API, configuration or build/deploy step, or learn a durable constraint or pattern. Bind new or changed source files in the concept's `resource:`/`sources:` frontmatter, and add a dated line to `okf/log.md`.
2. **Maintain:** `node C:/Users/kvlpv/.claude/skills/okf-maintain/scripts/maintain.js` (refreshes timestamps, the `## Not yet specified` frontier, `log.md` and `viz.html`).
3. **Lint:** `node C:/Users/kvlpv/.claude/skills/okf-lint/scripts/lint.js --drift`. It must report 0 errors and 0 warnings (use `;` rather than `&&` when chaining commands in Windows PowerShell).

<!-- okf-steering-version: 1.8.2 -->
