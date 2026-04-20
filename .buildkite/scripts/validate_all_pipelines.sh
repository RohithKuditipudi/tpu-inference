#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

# Assign the first argument to a local variable
RAW_FILES_TO_CHECK="${1:-}"
BUILDKITE_DIR=".buildkite"

# Pre-filter: Only include .yml or .yaml files located within the .buildkite/ directory
YAML_FILES_TO_CHECK=$(echo "$RAW_FILES_TO_CHECK" | grep -E "^\.buildkite/.*\.ya?ml$" || true)

# Early exit: If no YAML files were modified, skip validation
if [ -z "$YAML_FILES_TO_CHECK" ]; then
    echo "--- :crossed_fingers: No YAML changes detected. Skipping validation."
    exit 0
fi

# Initialize arrays and maps for uniqueness checks
VALIDATE_ARGS=()
declare -A PIPELINE_NAMES
declare -A CI_TARGETS

# 1. Discover spec directories
declare -a SPEC_DIRS=("quantization" "parallelism" "models" "features" "rl")
KERNEL_PARENT_DIR="$BUILDKITE_DIR/kernel_microbenchmarks"

echo "--- 📂 Discovering spec directories"
if [[ -d "$KERNEL_PARENT_DIR" ]]; then
    while IFS= read -r dir; do
        # Extract relative path from .buildkite/
        SPEC_DIRS+=("${dir#"$BUILDKITE_DIR"/}")
    done < <(find "$KERNEL_PARENT_DIR" -maxdepth 1 -mindepth 1 -type d)
fi

# 2. Perform Uniqueness Checks for pipeline-name and CI_TARGET
echo "--- 🔍 Checking for duplicate pipeline names and CI targets"
for folder in "${SPEC_DIRS[@]}"; do
    full_path="$BUILDKITE_DIR/$folder"
    [[ ! -d "$full_path" ]] && continue

    while IFS= read -r -d '' file; do
        # Only check files that define pipeline steps
        if grep -q "^[[:space:]]*steps:" "$file"; then
            # Extract # pipeline-name: value
            P_NAME_LINE=$(awk '/^[[:space:]]*#[[:space:]]*pipeline-name:/ {print $0; exit}' "$file")
            P_NAME=$(echo "${P_NAME_LINE#*:}" | xargs)

            # Extract CI_TARGET: value
            C_TARGET_RAW=$(grep -E "^[[:space:]]*CI_TARGET:" "$file" | head -1 || true)
            C_TARGET=$(echo "$C_TARGET_RAW" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\' | xargs)

            if [[ -n "$P_NAME" ]]; then
                if [[ -n "${PIPELINE_NAMES[$P_NAME]:-}" ]]; then
                    echo "+++ ❌ Error: Duplicate '# pipeline-name: $P_NAME' detected!"
                    echo "Conflict: $file and ${PIPELINE_NAMES[$P_NAME]}"
                    exit 1
                fi
                PIPELINE_NAMES["$P_NAME"]="$file"
            fi

            if [[ -n "$C_TARGET" ]]; then
                if [[ -n "${CI_TARGETS[$C_TARGET]:-}" ]]; then
                    echo "+++ ❌ Error: Duplicate 'CI_TARGET: $C_TARGET' detected!"
                    echo "Conflict: $file and ${CI_TARGETS[$C_TARGET]}"
                    exit 1
                fi
                CI_TARGETS["$C_TARGET"]="$file"
            fi
        fi
    done < <(find "$full_path" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done

# 3. Build validation arguments for changed files
echo "--- 📂 Preparing files for Buildkite validation"
while IFS= read -r file; do
    [ -z "$file" ] && continue

    if [ ! -f "$file" ]; then
        echo "Skipping deleted file: $file"
        continue
    fi

    echo "Adding to validation list: $file"
    VALIDATE_ARGS+=("--file" "$file")

done < <(echo "$YAML_FILES_TO_CHECK")

# 4. Final Buildkite Pipeline Validation
echo "--- 🔍 Validating changed YAML syntax"
if [ ${#VALIDATE_ARGS[@]} -gt 0 ]; then
    if ! bk pipeline validate "${VALIDATE_ARGS[@]}"; then
        echo "+++ ❌ Validation Failed"
        echo "Result: FAIL (Please fix the YAML syntax errors above)"
        exit 1
    else
        echo "+++ ✅ Validation Successful"
        echo "Result: SUCCESS"
        exit 0
    fi
else
    echo "--- :v: No existing YAML files found to validate."
    exit 0
fi
