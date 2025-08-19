#!/bin/bash

### Load ../dev.env (or override via ENV_FILE=/path/to/file)
script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ENV_FILE="${ENV_FILE:-"$script_dir/../dev.env"}"
if [[ -f "$ENV_FILE" ]]; then
  echo "Loading env from: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
else
  echo "No env file found at $ENV_FILE (continuing with current environment)"
fi

### Required / optional env
DB_NAME="${DB_NAME:-optuna_db}"
DB_USER="${DB_USER:-postgres}"
DB_PORT="${DB_PORT:-5432}"

# Local side defaults
LOCAL_DB_HOST="${LOCAL_DB_HOST:-localhost}"

# Remote side: do NOT hardcode; prefer DB_HOST, else LOGIN_NODE_HOSTNAME, else empty
REMOTE_DB_HOST="${DB_HOST:-${LOGIN_NODE_HOSTNAME:-}}"
if [[ -z "$REMOTE_DB_HOST" ]]; then
  echo "ERROR: REMOTE_DB_HOST is not set. Set DB_HOST or LOGIN_NODE_HOSTNAME in your env."
  exit 1
fi

# Password handling: prefer DB_PASS; else try ~/.pgpass lookups per host
DB_PASS="${DB_PASS:-}"

get_pgpass() {
  local host="$1" port="$2" user="$3"
  local pgpass="${PGPASSFILE:-$HOME/.pgpass}"
  if [[ -f "$pgpass" ]]; then
    awk -F: -v h="$host" -v p="$port" -v u="$user" '
      $1==h && $2==p && ($3=="*"||$3=="") && $4==u {print $5; found=1; exit}
      END{ if (!found) exit 1 }' "$pgpass" || true
  fi
}

# Resolve passwords for each side (can be identical)
LOCAL_DB_PASS="${LOCAL_DB_PASS:-${DB_PASS:-$(get_pgpass "$LOCAL_DB_HOST" "$DB_PORT" "$DB_USER")}}"
REMOTE_DB_PASS="${REMOTE_DB_PASS:-${DB_PASS:-$(get_pgpass "$REMOTE_DB_HOST" "$DB_PORT" "$DB_USER")}}"

timestamp="$(date +"%Y%m%d_%H%M%S")"
backup_dir="${BACKUP_DIR:-$script_dir/backups}"
mkdir -p "$backup_dir"

dump_custom() {
  # host port user db outfile
  local host="$1" port="$2" user="$3" db="$4" outfile="$5" pw="$6"
  echo "→ Dumping $db from $host:$port as $user → $outfile"
  PGPASSWORD="${pw:-}" pg_dump -h "$host" -p "$port" -U "$user" -d "$db" \
    -Fc -Z 9 --no-owner --no-privileges -f "$outfile"
}

ensure_db() {
  # host port user db pw
  local host="$1" port="$2" user="$3" db="$4" pw="$5"
  echo "→ Ensuring database '$db' exists on $host:$port"
  PGPASSWORD="${pw:-}" psql -h "$host" -p "$port" -U "$user" -tAc \
    "SELECT 1 FROM pg_database WHERE datname='${db}';" | grep -q 1 || \
  PGPASSWORD="${pw:-}" psql -h "$host" -p "$port" -U "$user" -c \
    "CREATE DATABASE ${db};"
}

restore_custom() {
  # host port user db infile pw
  local host="$1" port="$2" user="$3" db="$4" infile="$5" pw="$6"
  echo "→ Restoring $infile into $host:$port/$db"
  ensure_db "$host" "$port" "$user" "$db" "$pw"
  PGPASSWORD="${pw:-}" pg_restore -h "$host" -p "$port" -U "$user" \
    -d "$db" --clean --if-exists --no-owner --no-privileges "$infile"
}

pull() {
  # cluster -> local
  local out="$backup_dir/${DB_NAME}.from_remote_${timestamp}.dump"
  dump_custom "$REMOTE_DB_HOST" "$DB_PORT" "$DB_USER" "$DB_NAME" "$out" "$REMOTE_DB_PASS"
  restore_custom "$LOCAL_DB_HOST" "$DB_PORT" "$DB_USER" "$DB_NAME" "$out" "$LOCAL_DB_PASS"
  echo "✔ Pull complete: $REMOTE_DB_HOST → $LOCAL_DB_HOST"
}

push() {
  # local -> cluster
  local out="$backup_dir/${DB_NAME}.from_local_${timestamp}.dump"
  dump_custom "$LOCAL_DB_HOST" "$DB_PORT" "$DB_USER" "$DB_NAME" "$out" "$LOCAL_DB_PASS"
  restore_custom "$REMOTE_DB_HOST" "$DB_PORT" "$DB_USER" "$DB_NAME" "$out" "$REMOTE_DB_PASS"
  echo "✔ Push complete: $LOCAL_DB_HOST → $REMOTE_DB_HOST"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [pull|push|both]
  pull : sync cluster → local
  push : sync local   → cluster
  both : do pull then push (default)
Env of interest:
  DB_NAME, DB_USER, DB_PORT
  DB_HOST (remote) or LOGIN_NODE_HOSTNAME
  LOCAL_DB_HOST (default: localhost)
  DB_PASS (or LOCAL_DB_PASS / REMOTE_DB_PASS); ~/.pgpass also supported
  BACKUP_DIR (default: hpc/backups)
EOF
}

action="${1:-both}"
case "$action" in
  pull) pull ;;
  push) push ;;
  both) pull; push ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown action: $action"; usage; exit 1 ;;
esac
