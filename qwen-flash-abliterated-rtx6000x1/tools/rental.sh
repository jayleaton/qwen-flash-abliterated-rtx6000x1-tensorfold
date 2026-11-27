#!/usr/bin/env bash
# rental.sh STEP...: the RUNBOOK's RTX 6000 steps on one fresh host, from the prebuilt engine bundle; `all` runs
# them in order, one pack on disk at a time (the nvidia control pack, then the shipped pack), and writes
# $W/out/SUMMARY.txt. No compile: BUNDLE is the
# extracted `recipe kit` bundle (recipe/, engine/, kit/), which serves, gates, benches and soaks; this script's
# own recipe directory gives the weight checksums and receives the recorded gates/hashes.json.
#   W        work dir (default $HOME/rental); weights go to $W/models, logs and receipts to $W/out
#   BUNDLE   the bundle directory (default: the one bundle directory under $W)
#   MODEL, CONTROL  the shipped and control pack directories (default $W/models/abliterated, $W/models/control)
set -u
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
W=${W:-$HOME/rental}
BUNDLE=${BUNDLE:-$(find "$W" -maxdepth 1 -type d -name 'qwen-flash-abliterated-rtx6000x1-*' | head -1)}
R=$BUNDLE/recipe
E=$BUNDLE/engine
RECIPE=$BUNDLE/kit/bin/recipe
HOME_RECIPE=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-$W/models/abliterated}
CONTROL=${CONTROL:-$W/models/control}
OUT=$W/out
PORT=8090
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT" "$W/models"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/SUMMARY.txt"; }
die() { log "FAIL $*"; exit 1; }
toml() { python3 -c 'import sys, tomllib; d = tomllib.load(open(sys.argv[1], "rb")); print(d[sys.argv[2]][sys.argv[3]])' "$R/recipe.toml" "$@"; }

idle() { # idle WHAT: refuse to start WHAT while another process holds the GPU (a shared card voids every number)
  local apps; apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
  [ -z "$apps" ] || die "$1: the GPU is busy ($(echo "$apps" | tr '\n' ' ')); not starting"
}

serve() { # serve TAG TARGET MODEL [EXTRA...]: start a profile, wait for /v1/models, log ready time and device memory
  local tag=$1 target=$2 model=$3; shift 3
  idle "$tag"
  sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  local t0; t0=$(date +%s)
  "$RECIPE" serve --recipe "$R" --port "$PORT" "$target" "$E" "$model" -- "$@" > "$OUT/serve-$tag.log" 2>&1 &
  echo $! > "$OUT/serve.pid"
  for _ in $(seq 1 1800); do
    curl -sf "$URL/v1/models" > /dev/null 2>&1 && break
    kill -0 "$(cat "$OUT/serve.pid")" 2>/dev/null || { tail -30 "$OUT/serve-$tag.log"; die "$tag: server exited during boot"; }
    sleep 1
  done
  curl -sf "$URL/v1/models" > /dev/null || die "$tag: not ready after 30 min"
  log "$tag ready in $(($(date +%s) - t0)) s; device $(nvidia-smi --query-gpu=memory.used --format=csv,noheader); $(grep -m1 -o 'model [0-9.]* GiB.*' "$OUT/serve-$tag.log")"
}

stop() {
  local pid; pid=$(cat "$OUT/serve.pid" 2>/dev/null) || return 0
  pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  pkill -KILL -f "tensorfold-native serve" 2>/dev/null
  rm -f "$OUT/serve.pid"
}
trap stop EXIT

verified() { # verified DIR SHA256 BLOB: every large file's sha256 and every small file's git blob id match
  local dir=$1 sums=$2 blobs=$3
  [ -d "$dir" ] && (cd "$dir" && sha256sum -c --quiet "$sums" > /dev/null 2>&1) || return 1
  while read -r blob name; do
    [ "$(git hash-object "$dir/$name" 2>/dev/null)" = "$blob" ] || return 1
  done < "$blobs"
}

fetch() { # fetch REPO REV DIR SHA256 BLOB: download one pack at its revision unless it is there, then check every file
  local repo=$1 rev=$2 dir=$3 sums=$4 blobs=$5
  if ! verified "$dir" "$sums" "$blobs"; then
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$repo" --revision "$rev" --local-dir "$dir" > "$OUT/fetch-$(basename "$dir").log" 2>&1 \
      || die "download $repo"
    verified "$dir" "$sums" "$blobs" || die "$repo: a file differs from config/ checksums"
  fi
  log "fetch $repo@${rev:0:8}: $(wc -l < "$sums") sha256 and $(wc -l < "$blobs") blob checks pass ($(du -sh "$dir" | cut -f1))"
}

hashes_of() { # hashes_of GATE_JSON: the gate's hashes rows as {prompt: {greedy, sampled}}
  python3 - "$1" << 'EOF'
import json, sys
rows = [r for g in json.load(open(sys.argv[1])) if g["gate"] == "hashes" for r in g["rows"]]
out = {}
for r in rows:
    e = out.setdefault(r["prompt"], {"greedy": None, "sampled": []})
    (e.__setitem__("greedy", r["got"]) if r["mode"] == "greedy" else e["sampled"].append(r["got"]))
print(json.dumps(out))
EOF
}

record_hashes() { # record_hashes TAG: the hashes gate against placeholder rows, so it reports every prompt's hashes
  cp "$R/gates/hashes.json" "$OUT/hashes-before-$1.json"
  python3 - "$R/gates/hashes.json" << 'EOF'
import json, sys
table = json.load(open(sys.argv[1]))
zero = "0" * 12
table["expected"] = {"*": {name: {"greedy": zero, "sampled": [zero] * len(table["seeds"])} for name in table["prompts"]}}
open(sys.argv[1], "w").write(json.dumps(table, indent=1) + "\n")
EOF
  "$RECIPE" gate --recipe "$R" --tree "$E" --only hashes --output "$OUT/hashes-$1.json" > "$OUT/hashes-$1.txt" 2>&1
  cp "$OUT/hashes-before-$1.json" "$R/gates/hashes.json"
}

cold() { # cold SIZE TAG: one unsalted cold prompt (bench_cache, 16 tokens), prints the reply's token sha256
  python3 "$E/tools/bench_cache.py" "$URL" "$(toml recipe served_name)" --tokens 16 --repeats 1 --streams 1 \
    --require-cold --prompt-file "$R/prompts/ids-$1.json" --output "$OUT/cold-$2-$1.json" > "$OUT/cold-$2-$1.txt" 2>&1
  python3 -c 'import json, sys; r = json.load(open(sys.argv[1]))["rows"][0]; print(r.get("token_sha256") or r.get("token_sha"), round(r["prompt_tokens"] / r["ttft_s"]))' "$OUT/cold-$2-$1.json"
}

step_check() {
  local gpu mem drv py disk ram
  mkdir -p "$(dirname "$MODEL")"
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  py=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  disk=$(df -BG --output=avail "$(dirname "$MODEL")" | tail -1 | tr -dc 0-9)
  ram=$(awk '/MemTotal/{print int($2 / 1048576)}' /proc/meminfo)
  log "host: $gpu, $mem MiB, driver $drv, python $py, $disk GiB free for weights, $ram GiB RAM, $(nproc) CPUs ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs))"
  log "power: limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader), max $(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader) (the old tp1 reference ran at 600 W)"
  case "$gpu" in *"RTX PRO 6000"*) ;; *) die "not an RTX PRO 6000: $gpu" ;; esac
  [ "$mem" -ge 95000 ] || die "card memory $mem MiB < 95000"
  [ "${drv%%.*}" -ge 580 ] || die "driver $drv: CUDA 13.0 needs 580 or later"
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "python $py: the kit needs 3.11+"
  [ "$disk" -ge 130 ] || die "$disk GiB free for weights: one pack at a time takes up to 126 GiB"
  [ "$ram" -ge 120 ] || log "warning: $ram GiB RAM; the PLE table (47.7 GiB) is page cache, so boots will read from disk"
  command -v hf > /dev/null || python3 -m pip install -q --user -U "huggingface_hub[hf_transfer]" || die "install huggingface_hub"
  "$RECIPE" kit --verify "$BUNDLE" > "$OUT/bundle-verify.txt" 2>&1 || die "bundle: $(tail -3 "$OUT/bundle-verify.txt")"
  log "bundle $(basename "$BUNDLE") verifies; $(grep '^binary engine/zig-out/native/bin' "$BUNDLE/MANIFEST.txt")"
}

step_fetch_control() {
  fetch nvidia/Qwen3.8-Flash-Next-NVFP4 "$(cat "$HOME_RECIPE/config/control.rev")" "$CONTROL" \
    "$HOME_RECIPE/config/control.sha256" "$HOME_RECIPE/config/control.blob"
}

step_control() {
  # R1a and the control half of R3: the nvidia pack against the old recipe's one-GPU reference (code 4e759dbd859c,
  # chat 764da77fbd3f, cold 8K fa345c10f631; code x1 291 tok/s, cold 8K 8,818 tok/s at 600 W). A difference is a
  # finding to report, not a failure of this script.
  serve control default "$CONTROL"
  record_hashes control
  log "control: $(hashes_of "$OUT/hashes-control.json" | python3 -c 'import json, sys; d = json.load(sys.stdin); print(" ".join(k + " " + v["greedy"][:12] for k, v in d.items()))') (reference code 4e759dbd859c chat 764da77fbd3f)"
  log "control: cold 8K $(cold 8192 control) (hash, tok/s; reference fa345c10f631, 8,818 tok/s)"
  stop
  idle bench-control
  "$RECIPE" bench --recipe "$R" default optimised "$E" "$CONTROL" --out "$OUT/bench-control" > "$OUT/bench-control.txt" 2>&1 \
    || die "bench, control pack: $(tail -3 "$OUT/bench-control.txt" | tr '\n' ' ')"
  log "bench default vs optimised, control pack: $(grep -m1 'against default' "$OUT/bench-control.txt") (old tp1: code x1 291, cold 8K 8,818)"
}

step_drop_control() { # the disk holds one pack: the control pack goes before the shipped one comes
  [ -d "$CONTROL" ] && rm -rf -- "$CONTROL"
  log "control pack removed; $(df -BG --output=avail "$(dirname "$MODEL")" | tail -1 | xargs) free"
}

step_fetch() {
  fetch "$(toml recipe model)" "$(toml weights rev)" "$MODEL" "$HOME_RECIPE/config/weights.sha256" "$HOME_RECIPE/config/weights.blob"
}

step_fit() {
  serve fit-default default "$MODEL"; stop
  serve fit-agent agent "$MODEL"; stop
}

step_hashes() {
  # R1b: the shipped pack's own hashes, recorded into the bundle's gates/hashes.json and this recipe's
  serve hashes default "$MODEL"
  record_hashes record
  local c8 c32; c8=$(cold 8192 shipped); c32=$(cold 32768 shipped)
  python3 - "$R/gates/hashes.json" "$(hashes_of "$OUT/hashes-record.json")" "${c8%% *}" "${c32%% *}" << 'EOF'
import json, re, sys
path, rows, c8, c32 = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4]
table = json.load(open(path))
table["expected"] = {"*": rows}
for size, digest in (("8192", c8), ("32768", c32)):
    if re.fullmatch(r"[0-9a-f]{64}", digest):
        table["cold"][size]["hash"] = digest
open(path, "w").write(json.dumps(table, indent=1) + "\n")
EOF
  cp "$R/gates/hashes.json" "$OUT/hashes.json"
  cp "$R/gates/hashes.json" "$HOME_RECIPE/gates/hashes.json"
  log "shipped: recorded gates/hashes.json (commit it from $HOME_RECIPE); cold 8K $c8, 32K $c32 (hash, tok/s)"
  stop
}

step_gate() {
  for target in default agent; do
    serve "gate-$target" "$target" "$MODEL"
    "$RECIPE" gate --recipe "$R" --tree "$E" "$target" --output "$OUT/gate-$target.json" --receipt "$OUT/gate-$target-receipt.json" > "$OUT/gate-$target.txt" 2>&1
    local rc=$?
    log "gate $target: $(tail -1 "$OUT/gate-$target.txt")"
    [ $rc = 0 ] || log "gate $target failed (rc $rc); continuing to the next profile"
    stop
  done
}

step_bench() {
  idle bench
  "$RECIPE" bench --recipe "$R" default optimised "$E" "$MODEL" --out "$OUT/bench-shipped" > "$OUT/bench-shipped.txt" 2>&1 \
    || die "bench, shipped pack: $(tail -3 "$OUT/bench-shipped.txt" | tr '\n' ' ')"
  log "bench default vs optimised, shipped pack: $(grep -m1 'against default' "$OUT/bench-shipped.txt")"
}

step_soak() {
  idle soak
  "$RECIPE" soak --recipe "$R" default "$E" "$MODEL" --out "$OUT/soak-default" > "$OUT/soak-default.txt" 2>&1 \
    || die "soak: $(tail -3 "$OUT/soak-default.txt" | tr '\n' ' ')"
  log "soak default: $(tail -1 "$OUT/soak-default.txt")"
}

[ -d "$R" ] && [ -x "$RECIPE" ] || die "no bundle at '$BUNDLE' (set BUNDLE=)"
ALL=(check fetch_control control drop_control fetch fit hashes gate bench soak)
steps=("$@"); [ "${steps[0]:-}" = all ] && steps=("${ALL[@]}")
[ ${#steps[@]} -gt 0 ] || { echo "usage: rental.sh all | ${ALL[*]}"; exit 2; }
for s in "${steps[@]}"; do
  declare -F "step_$s" > /dev/null || die "unknown step $s"
  log "== $s"; "step_$s"
done
log "done: ${steps[*]}"
