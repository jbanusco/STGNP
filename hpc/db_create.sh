#!/bin/bash

### Locate and load ../dev.env relative to this script (root/hpc/script.sh -> root/dev.env)
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

### Defaults (all overridable by env)
DATA_PATH="${DATA_PATH:-/data/bdip2/jbanusco}"
SINGULARITY_FOLDER="${SINGULARITY_FOLDER:-$DATA_PATH/SingularityImages}"
SINGULARITY_IMG="${SINGULARITY_IMG:-$SINGULARITY_FOLDER/postgres_latest.sif}"

DB_NAME="${DB_NAME:-optuna_db}"
DB_USER="${DB_USER:-postgres}"
DB_PORT="${DB_PORT:-5432}"

# Do NOT hardcode a host/IP by default:
# Use DB_HOST if provided; else LOGIN_NODE_HOSTNAME if provided; else leave empty and warn.
DB_HOST="${DB_HOST:-${LOGIN_NODE_HOSTNAME:-}}"
if [[ -z "${DB_HOST}" ]]; then
  echo "WARNING: DB_HOST is not set (and LOGIN_NODE_HOSTNAME not provided). TCP psql connection step will be skipped."
fi

DB_PATH="${DB_PATH:-$DATA_PATH/postgres_data}"
PG_LOCK_DIR="${PG_LOCK_DIR:-$DB_PATH/pg_lock}"
ALLOW_SUBNETS="${ALLOW_SUBNETS:-10.0.0.0/8}"  # override in env as needed

mkdir -p "$DB_PATH" "$PG_LOCK_DIR"

### Resolve postgres password:
# 1) Use DB_PASS if provided in env
# 2) Else try to extract from ~/.pgpass for (DB_HOST:DB_PORT:*:DB_USER)
DB_PASS="${DB_PASS:-}"
if [[ -z "${DB_PASS}" ]]; then
  PGPASS_FILE="${PGPASSFILE:-$HOME/.pgpass}"
  if [[ -f "$PGPASS_FILE" ]]; then
    # Try exact host match
    DB_PASS="$(awk -F: -v h="$DB_HOST" -v p="$DB_PORT" -v u="$DB_USER" \
      '$1==h && $2==p && ($3=="*"||$3=="") && $4==u {print $5; found=1} END{if(!found) exit 1}' "$PGPASS_FILE" || true)"
    # Fallback to localhost entry if still empty
    if [[ -z "$DB_PASS" ]]; then
      DB_PASS="$(awk -F: -v p="$DB_PORT" -v u="$DB_USER" \
        '$1=="localhost" && $2==p && ($3=="*"||$3=="") && $4==u {print $5; exit 0}' "$PGPASS_FILE" || true)"
    fi
  fi
fi
# DB_PASS may still be empty; CREATE ROLE will fail if so—warn:
if [[ -z "${DB_PASS}" ]]; then
  echo "WARNING: DB_PASS is empty (no env and no matching ~/.pgpass). User creation will use an empty password."
fi

### Helper to run inside the container with data bind mounted
sing() {
  singularity exec --bind "$DB_PATH:/var/lib/postgresql/data" "$SINGULARITY_IMG" "$@"
}

### Initialize data dir
echo "Checking if PostgreSQL data directory is initialized at $DB_PATH..."
if [[ ! -f "$DB_PATH/PG_VERSION" ]]; then
  echo "Initializing PostgreSQL database..."
  sing bash -c 'export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8; initdb --encoding=UTF8 --locale=en_US.UTF-8 -D /var/lib/postgresql/data'
else
  echo "PostgreSQL data directory already initialized."
fi

### Start server if not running
echo "Ensuring PostgreSQL is running..."
if ! sing pg_ctl status >/dev/null 2>&1; then
  sing bash -c "pg_ctl -D /var/lib/postgresql/data -o \"-c unix_socket_directories=$PG_LOCK_DIR\" start -l /var/lib/postgresql/data/postgres.log"
  echo "PostgreSQL started."
else
  echo "PostgreSQL already running."
fi

### Create role if needed
user_exists="$(sing bash -c "export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8; psql -h '$PG_LOCK_DIR' -d template1 -tAc \"SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}';\"")"
if [[ "$user_exists" != "1" ]]; then
  echo "Creating user '${DB_USER}'..."
  sing bash -c "export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8; psql -h '$PG_LOCK_DIR' -d template1 -c \"CREATE ROLE ${DB_USER} WITH SUPERUSER CREATEDB CREATEROLE LOGIN PASSWORD '${DB_PASS}';\""
else
  echo "User '${DB_USER}' already exists."
fi

### Create database if needed
db_exists="$(sing bash -c "export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8; psql -U ${DB_USER} -h '$PG_LOCK_DIR' -tAc \"SELECT 1 FROM pg_database WHERE datname='${DB_NAME}';\"")"
if [[ "$db_exists" != "1" ]]; then
  echo "Creating database '${DB_NAME}'..."
  sing bash -c "export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8; psql -U ${DB_USER} -h '$PG_LOCK_DIR' -c \"CREATE DATABASE ${DB_NAME};\""
else
  echo "Database '${DB_NAME}' already exists."
fi

### Configure listen_addresses (avoid duplicates)
if ! sing bash -c "grep -q \"^listen_addresses = '\\*'\" /var/lib/postgresql/data/postgresql.conf"; then
  echo "Enabling listen_addresses='*'..."
  sing bash -c "echo \"listen_addresses = '*'\" >> /var/lib/postgresql/data/postgresql.conf"
fi

### Configure pg_hba.conf for allowed subnets (avoid duplicates)
for net in $ALLOW_SUBNETS; do
  rule="host    all    ${DB_USER}    ${net}    md5"
  if ! sing bash -c "grep -qE \"^host\\s+all\\s+${DB_USER}\\s+${net//\//\\/}\\s+md5\\s*$\" /var/lib/postgresql/data/pg_hba.conf"; then
    echo "Allowing subnet ${net} in pg_hba.conf"
    sing bash -c "echo \"$rule\" >> /var/lib/postgresql/data/pg_hba.conf"
  fi
done

### Restart to apply changes
echo "Restarting PostgreSQL to apply configuration..."
sing bash -c "pg_ctl -D /var/lib/postgresql/data restart"

### Example interactive psql (TCP) — can be commented out if undesired
echo "Connecting with psql to ${DB_HOST}:${DB_PORT}/${DB_NAME} as ${DB_USER}..."
sing psql -U "${DB_USER}" -h "${DB_HOST}" -p "${DB_PORT}" -d "${DB_NAME}"

### Background run helper (optional)
# nohup sing bash -c "pg_ctl -D /var/lib/postgresql/data -o \"-c unix_socket_directories=${PG_LOCK_DIR}\" start -l /var/lib/postgresql/data/postgres.log" > pgserver.log 2>&1 &
