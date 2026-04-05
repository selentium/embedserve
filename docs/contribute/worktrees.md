# Worktrees

The repository includes helpers for creating isolated sibling worktrees on new branches.

## Create a Worktree

Basic usage:

```bash
make worktree-create WORKTREE=agent-a
```

Default behavior:

- base ref `HEAD`
- path `../embedserve-agent-a`
- copy `.env` when it exists
- bootstrap the new worktree with `make bootstrap-dev`

## Common Variants

Different base:

```bash
make worktree-create WORKTREE=bugfix BASE=main
```

Different path:

```bash
make worktree-create WORKTREE=perf WORKTREE_PATH=../scratch/embedserve-perf
```

Skip setup or `.env` copy:

```bash
make worktree-create WORKTREE=docs SETUP=0 COPY_ENV=0
```

## Remove a Worktree

Remove a clean worktree but keep the branch:

```bash
make worktree-remove WORKTREE=agent-a
```

Force-remove a dirty worktree and delete the branch:

```bash
make worktree-remove WORKTREE=agent-a FORCE=1 DELETE_BRANCH=1
```

## When To Use Worktrees

- parallel feature work
- risky refactors
- benchmark or load-test branches
- keeping documentation edits isolated from runtime changes
