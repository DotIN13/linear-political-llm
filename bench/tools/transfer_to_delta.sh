#!/usr/bin/env bash
#
# Move the whole linear-political-llm directory from Midway3 to DeltaAI over Globus.
#
#   ./bench/tools/transfer_to_delta.sh
#
# You handle the browser logins; the script handles everything around them. It is
# safe to re-run: --sync-level checksum means a second run transfers only what is
# missing or different, so an interrupted 20 GB copy resumes rather than restarts.
#
# Nothing is submitted until every precondition passes and you have seen the two
# paths printed back to you.
#
# Override any of these if the defaults are wrong:
#   SRC_SEARCH / DST_SEARCH   what to search for when resolving collections
#   SRC_ID / DST_ID           skip the search entirely, give UUIDs
#   SRC_PATH / DST_PATH       the directories
#   ASSUME_YES=1              do not prompt before submitting
#   DRY_RUN=1                 run every check, print the command, submit nothing

set -uo pipefail

SRC_SEARCH="${SRC_SEARCH:-midway3}"
DST_SEARCH="${DST_SEARCH:-delta}"
SRC_PATH="${SRC_PATH:-/project/jevans/tzhang3/dotty-project/linear-political-llm/}"
DST_PATH="${DST_PATH:-/work/nvme/bifr/tzhang30/lpl/incoming/midway-lpl/}"
LABEL="${LABEL:-lpl midway3 to deltaai $(date +%Y-%m-%d)}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# --- 0. the CLI ---------------------------------------------------------------
step "0. Globus CLI"
# On Midway, PYTHONUSERBASE points at /scratch/.../.local, so
# `pip install --user` drops the binary in $PYTHONUSERBASE/bin, which is NOT
# on PATH by default (PATH carries ~/.local/bin instead). Look in the usual
# user locations before giving up, so an installed-but-not-on-PATH CLI just
# works.
_add_site_packages() {
  # $1 = <prefix> (the dir holding bin/ and lib/); puts its
  # lib/python3*/site-packages on PYTHONPATH if globus_cli lives there.
  local _prefix="$1" _sp
  for _sp in "$_prefix"/lib/python3*/site-packages; do
    if [ -d "$_sp/globus_cli" ]; then
      export PYTHONPATH="${_sp}${PYTHONPATH:+:$PYTHONPATH}"
      break
    fi
  done
}
if [ -n "${GLOBUS_BIN:-}" ] && [ -x "$GLOBUS_BIN" ]; then
  export PATH="$(dirname "$GLOBUS_BIN"):$PATH"
  _add_site_packages "$(dirname "$(dirname "$GLOBUS_BIN")")"
fi
for _d in "${PYTHONUSERBASE:-}/bin" "$HOME/.local/bin" \
          "/scratch/midway3/${USER:-}/.local/bin" \
          "${SCRATCH:-}/midway3/${USER:-}/.local/bin"; do
  if [ -n "$_d" ] && [ -x "$_d/globus" ] && ! command -v globus >/dev/null 2>&1; then
    export PATH="$_d:$PATH"
    # The binary alone is not enough: its module lives under
    # <prefix>/lib/python3*/site-packages, which python only sees via
    # PYTHONUSERBASE (set in a normal Midway shell, but not guaranteed).
    # Put it on PYTHONPATH so the CLI works even with PYTHONUSERBASE unset.
    _add_site_packages "$(dirname "$_d")"
  fi
done
unset _d
unset -f _add_site_packages 2>/dev/null || true
if ! command -v globus >/dev/null 2>&1; then
  die "globus not found. Install it, then re-run:

    pipx install globus-cli        # preferred, keeps it isolated
    # or
    python3 -m pip install --user globus-cli

  On Midway, '--user' installs to \$PYTHONUSERBASE/bin
  (/scratch/midway3/\$USER/.local/bin), which is not on PATH. Either re-run
  with it on PATH:

    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    ./bench/tools/transfer_to_delta.sh

  or point at the binary directly:

    GLOBUS_BIN=/scratch/midway3/\$USER/.local/bin/globus ./bench/tools/transfer_to_delta.sh"
fi
echo "  $(globus version 2>&1 | head -1)"

# --- 1. login -----------------------------------------------------------------
step "1. Login"
if globus whoami >/dev/null 2>&1; then
  echo "  already logged in as $(globus whoami 2>/dev/null)"
else
  echo "  opening a browser login..."
  globus login || die "login failed"
  echo "  logged in as $(globus whoami 2>/dev/null)"
fi

# --- 2. resolve the two collections -------------------------------------------
# Not hardcoded as UUIDs on purpose: NCSA publishes separate guides for Delta and
# DeltaAI, and /work/nvme is the DeltaAI filesystem. A UUID copied from the wrong
# guide is the single most likely way to waste a 20 GB transfer, so we search,
# print what we found, and prove the path is visible before sending anything.
resolve() {
  # --format unix is deliberately NOT used here. The unix printer emits the
  # raw endpoint dicts with keys sorted alphabetically, so the first column
  # is not the ID. JSON plus a python3 one-liner extracts IDs reliably.
  local term="$1" label="$2" preset="$3"
  if [ -n "$preset" ]; then echo "$preset"; return 0; fi
  local json out
  json=$(globus endpoint search "$term" --filter-scope all --limit 20 -F json 2>/dev/null) || return 1
  out=$(printf '%s' "$json" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for ep in data.get('DATA', []):
    eid = ep.get('id', '')
    name = ep.get('display_name') or ep.get('canonical_name') or ''
    if eid:
        print(eid + '\t' + str(name))
") || return 1
  [ -n "$out" ] || return 1
  local n; n=$(printf '%s\n' "$out" | wc -l | tr -d ' ')
  if [ "$n" = 1 ]; then printf '%s\n' "$out" | awk -F'\t' '{print $1}'; return 0; fi
  {
    warn ""
    warn "More than one collection matches '$term' for the $label side:"
    printf '%s\n' "$out" | nl -w4 -s'  ' >&2
    warn ""
    warn "Pick one and re-run with its UUID, e.g.:"
    warn "    ${label^^}_ID=<uuid> $0"
  }
  return 2
}

step "2. Collections"
SRC_ID="${SRC_ID:-$(resolve "$SRC_SEARCH" src "${SRC_ID:-}")}" || {
  [ $? = 2 ] && exit 1
  die "no collection matched '$SRC_SEARCH'. Set SRC_ID=<uuid> and re-run."
}
DST_ID="${DST_ID:-$(resolve "$DST_SEARCH" dst "${DST_ID:-}")}" || {
  [ $? = 2 ] && exit 1
  die "no collection matched '$DST_SEARCH'. Set DST_ID=<uuid> and re-run."
}
echo "  source      $SRC_ID"
globus endpoint show "$SRC_ID" 2>/dev/null | head -5 | sed 's/^/              /'
echo "  destination $DST_ID"
globus endpoint show "$DST_ID" 2>/dev/null | head -5 | sed 's/^/              /'

# --- 3. prove both paths are actually reachable -------------------------------
# This is where ConsentRequired shows up on mapped collections, and it is much
# better to hit it here than three seconds into a transfer.
check_path() {
  local id="$1" path="$2" side="$3" out rc
  out=$(globus ls "$id:$path" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    echo "  $side ok -- $(printf '%s\n' "$out" | wc -l | tr -d ' ') entries"
    return 0
  fi
  if printf '%s' "$out" | grep -qi 'consent'; then
    warn ""
    warn "The $side collection needs a one-off consent. Globus printed the exact"
    warn "command to run -- it looks like 'globus session consent ...'. Here is its"
    warn "message in full:"
    printf '%s\n' "$out" | sed 's/^/    /' >&2
    warn ""
    warn "Run that command, then re-run this script. Nothing has been transferred."
    exit 1
  fi
  warn ""
  warn "Cannot list $side path:"
  warn "    $id:$path"
  printf '%s\n' "$out" | sed 's/^/    /' >&2
  if [ "$side" = destination ]; then
    warn ""
    warn "If this says the path does not exist, create it first:"
    warn "    mkdir -p $DST_PATH        # on DeltaAI"
    warn "If it says the collection cannot see /work/nvme, you have Delta rather"
    warn "than DeltaAI -- search for the DeltaAI collection and pass DST_ID."
  fi
  exit 1
}
step "3. Paths"
echo "  source      $SRC_PATH"
check_path "$SRC_ID" "$SRC_PATH" source
echo "  destination $DST_PATH"
check_path "$DST_ID" "$DST_PATH" destination

# --- 4. confirm ---------------------------------------------------------------
step "4. About to submit"
cat <<EOF
  from   $SRC_ID
         $SRC_PATH
  to     $DST_ID
         $DST_PATH
  label  $LABEL

  recursive, --sync-level checksum, --verify-checksum, --preserve-timestamp.

  Re-running this script later is safe and cheap: checksum sync sends only what
  is missing or different, so an interrupted transfer resumes.
EOF
if [ "${DRY_RUN:-0}" = 1 ]; then
  bold "
DRY_RUN=1 -- would run:"
  echo "  globus transfer -r --sync-level checksum --verify-checksum \\"
  echo "      --preserve-timestamp --label \"$LABEL\" \\"
  echo "      $SRC_ID:$SRC_PATH $DST_ID:$DST_PATH"
  exit 0
fi
if [ "${ASSUME_YES:-0}" != 1 ]; then
  printf '\n  Proceed? [y/N] '
  read -r reply < /dev/tty
  case "$reply" in [yY]*) ;; *) echo "  nothing submitted."; exit 0 ;; esac
fi

# --- 5. submit ----------------------------------------------------------------
step "5. Submitting"
OUT=$(globus transfer -r \
        --sync-level checksum \
        --verify-checksum \
        --preserve-timestamp \
        --label "$LABEL" \
        "$SRC_ID:$SRC_PATH" "$DST_ID:$DST_PATH" 2>&1)
RC=$?
printf '%s\n' "$OUT" | sed 's/^/  /'
[ $RC -eq 0 ] || die "
submit failed. Nothing is running."

TASK=$(printf '%s' "$OUT" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)
[ -n "$TASK" ] || { warn "submitted, but could not parse a task id from the output"; exit 0; }

step "6. Running"
cat <<EOF
  task     $TASK
  watch    https://app.globus.org/activity/$TASK
  wait     globus task wait $TASK
  cancel   globus task cancel $TASK

  You can close this terminal -- the transfer runs on Globus, not here.
EOF

cat <<'EOF'

== When it finishes ==

  1. Compare the byte count Globus reports against the inventory total.

  2. Move the plain directories into the layout. Same filesystem, so these are
     renames -- instant, and they cost no space:

       cd /work/nvme/bifr/tzhang30
       mv lpl/incoming/midway-lpl/items         lpl/shared/items
       mv lpl/incoming/midway-lpl/runs          lpl/shared/runs
       mv lpl/incoming/midway-lpl/judge_cache   lpl/shared/judge_cache
       mv lpl/incoming/midway-lpl/conversations lpl/shared/conversations
       mv lpl/incoming/midway-lpl/datasets/*    datasets/

  3. Stage the images. THIS STEP IS NOT OPTIONAL AND IT IS NOT A MOVE.
     The resized files are named by a content hash; the items files call them by
     COCO record name; the hash is not derivable from the name. Skip this and
     every image lookup misses silently -- a missing image does not raise, it
     just produces an answer with fewer pixels behind it.

       cd lpl/wt/dev && source ../../env.sh
       python -m bench.tools.stage_images \
         --items  ../../shared/items/explore_bucket_v1.jsonl \
         --source ../../incoming/midway-lpl/results/token_scoring/qwen3_vl/lvis/_resized_images_800 \
         --dest   ../../shared/images \
         --link

  4. Verify, on its own. This is what catches a staging run that did nothing:

       python -m bench.tools.stage_images \
         --items ../../shared/items/explore_bucket_v1.jsonl \
         --dest  ../../shared/images --verify-only

  5. Keep lpl/incoming/ until a pilot has actually run. It costs nothing against
     3.4 PB and it makes a mistake in step 2 recoverable without a second copy.
EOF
