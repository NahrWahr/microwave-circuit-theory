#!/usr/bin/env bash
# Render all scenes and concatenate them into a single video.
# Usage: ./render_all.sh [quality]   quality ∈ {l,m,h,k}  (default: h)

set -euo pipefail

QUALITY="${1:-h}"
case "$QUALITY" in
    l) QDIR="480p15" ;;
    m) QDIR="720p30" ;;
    h) QDIR="1080p60" ;;
    k) QDIR="2160p60" ;;
    *) echo "Unknown quality '$QUALITY' (use l|m|h|k)"; exit 1 ;;
esac

cd "$(dirname "$0")"
ROOT="$(pwd)"
OUT_DIR="$ROOT/media/videos"
FINAL="$ROOT/microwave_circuit_theory.mp4"

SCENES=(
    "scene_01_transmission_line.py:TransmissionLine"
    "scene_02_reflection_and_impedance.py:ReflectionAndImpedance"
    "scene_03_two_port_and_s_params.py:TwoPortAndSParams"
    "scene_04_signal_flow_and_gain.py:SignalFlowAndGain"
    "scene_04b_bilinear_transformations.py:BilinearTransformations"
    "scene_05_stability_and_unilateral.py:StabilityAndUnilateral"
    "scene_06_masons_u.py:MasonsU"
    "scene_07_nonlinear_and_noise.py:NonlinearAndNoise"
)

LIST="$(mktemp --suffix=.txt)"
trap 'rm -f "$LIST"' EXIT

echo "Rendering ${#SCENES[@]} scenes at quality=$QUALITY ($QDIR)..."
for entry in "${SCENES[@]}"; do
    file="${entry%%:*}"
    cls="${entry##*:}"
    echo ">>> $file :: $cls"
    uv run manim -q"$QUALITY" "$file" "$cls"
    stem="${file%.py}"
    mp4="$OUT_DIR/$stem/$QDIR/$cls.mp4"
    [[ -f "$mp4" ]] || { echo "Missing: $mp4"; exit 1; }
    printf "file '%s'\n" "$mp4" >> "$LIST"
done

echo ">>> Concatenating -> $FINAL"
ffmpeg -y -f concat -safe 0 -i "$LIST" -c copy "$FINAL"
echo "Done: $FINAL"
