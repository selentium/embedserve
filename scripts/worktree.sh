#!/usr/bin/env bash

set -euo pipefail

die() {
	echo "Error: $*" >&2
	exit 1
}

is_truthy() {
	case "${1:-0}" in
		1|true|TRUE|yes|YES|on|ON)
			return 0
			;;
		*)
			return 1
			;;
	esac
}

repo_root() {
	git rev-parse --show-toplevel
}

default_worktree_path() {
	local root
	root="$(repo_root)"
	printf '%s/%s-%s\n' "$(dirname "$root")" "$(basename "$root")" "$WORKTREE_NAME"
}

canonicalize_new_path() {
	local input_path parent basename_value
	input_path="$1"
	parent="$(dirname "$input_path")"
	basename_value="$(basename "$input_path")"

	if [ ! -d "$parent" ]; then
		die "Parent directory does not exist: $parent"
	fi

	(
		cd "$parent"
		printf '%s/%s\n' "$(pwd -P)" "$basename_value"
	)
}

canonicalize_existing_path() {
	local input_path
	input_path="$1"

	if [ ! -d "$input_path" ]; then
		die "Worktree path does not exist: $input_path"
	fi

	(
		cd "$input_path"
		pwd -P
	)
}

is_registered_worktree() {
	local target_path worktree_path
	target_path="$1"

	while IFS= read -r worktree_path; do
		if [ "$worktree_path" = "$target_path" ]; then
			return 0
		fi
	done <<EOF
$(git worktree list --porcelain | sed -n 's/^worktree //p')
EOF

	return 1
}

ensure_branch_does_not_exist() {
	if git show-ref --verify --quiet "refs/heads/$WORKTREE_NAME"; then
		die "Branch already exists: $WORKTREE_NAME"
	fi
}

ensure_base_ref_exists() {
	if ! git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null; then
		die "Base ref not found: $BASE_REF"
	fi
}

ensure_worktree_is_clean() {
	local target_path
	target_path="$1"

	if [ -n "$(git -C "$target_path" status --porcelain --untracked-files=normal)" ]; then
		die "Worktree has uncommitted changes: $target_path"
	fi
}

copy_env_if_requested() {
	local root source_env target_env
	root="$(repo_root)"
	source_env="$root/.env"
	target_env="$WORKTREE_PATH/.env"

	if is_truthy "${COPY_ENV:-0}" && [ -f "$source_env" ] && [ ! -e "$target_env" ]; then
		cp "$source_env" "$target_env"
	fi
}

run_setup_if_requested() {
	if ! is_truthy "${SETUP:-0}"; then
		return
	fi

	if [ -z "${SETUP_COMMAND:-}" ]; then
		die "SETUP=1 requires SETUP_COMMAND"
	fi
	export WORKTREE_PATH
	eval "$SETUP_COMMAND"
}

rollback_created_worktree() {
	if [ -n "${WORKTREE_PATH:-}" ] && [ -d "$WORKTREE_PATH" ] && is_registered_worktree "$WORKTREE_PATH"; then
		git worktree remove --force "$WORKTREE_PATH" >/dev/null 2>&1 || true
	fi

	if [ -n "${WORKTREE_NAME:-}" ] && git show-ref --verify --quiet "refs/heads/$WORKTREE_NAME"; then
		git branch -D "$WORKTREE_NAME" >/dev/null 2>&1 || true
	fi

	git worktree prune >/dev/null 2>&1 || true
}

create_worktree() {
	if [ -z "${WORKTREE_NAME:-}" ]; then
		die "WORKTREE is required"
	fi

	BASE_REF="${BASE_REF:-HEAD}"
	WORKTREE_PATH="${WORKTREE_PATH:-$(default_worktree_path)}"
	WORKTREE_PATH="$(canonicalize_new_path "$WORKTREE_PATH")"

	if [ -e "$WORKTREE_PATH" ]; then
		die "Worktree path already exists: $WORKTREE_PATH"
	fi

	ensure_branch_does_not_exist
	ensure_base_ref_exists

	git worktree add -b "$WORKTREE_NAME" "$WORKTREE_PATH" "$BASE_REF"
	if ! copy_env_if_requested; then
		rollback_created_worktree
		return 1
	fi
	if ! run_setup_if_requested; then
		rollback_created_worktree
		return 1
	fi

	echo "Created worktree $WORKTREE_NAME at $WORKTREE_PATH"
}

delete_branch_if_requested() {
	if ! is_truthy "${DELETE_BRANCH:-0}"; then
		return
	fi

	if ! git show-ref --verify --quiet "refs/heads/$WORKTREE_NAME"; then
		die "Branch not found: $WORKTREE_NAME"
	fi

	if is_truthy "${FORCE:-0}"; then
		git branch -D "$WORKTREE_NAME"
	else
		git branch -d "$WORKTREE_NAME"
	fi
}

remove_worktree() {
	if [ -z "${WORKTREE_NAME:-}" ]; then
		die "WORKTREE is required"
	fi

	WORKTREE_PATH="${WORKTREE_PATH:-$(default_worktree_path)}"
	WORKTREE_PATH="$(canonicalize_existing_path "$WORKTREE_PATH")"

	if ! is_registered_worktree "$WORKTREE_PATH"; then
		die "Not a registered worktree: $WORKTREE_PATH"
	fi

	if ! is_truthy "${FORCE:-0}"; then
		ensure_worktree_is_clean "$WORKTREE_PATH"
	fi

	if is_truthy "${FORCE:-0}"; then
		git worktree remove --force "$WORKTREE_PATH"
	else
		git worktree remove "$WORKTREE_PATH"
	fi

	git worktree prune
	delete_branch_if_requested

	echo "Removed worktree $WORKTREE_NAME from $WORKTREE_PATH"
}

main() {
	local command
	command="${1:-}"

	case "$command" in
		create)
			create_worktree
			;;
		remove)
			remove_worktree
			;;
		*)
			die "Usage: bash scripts/worktree.sh [create|remove]"
			;;
	esac
}

main "$@"
