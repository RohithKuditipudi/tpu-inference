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
# This automatically ignores .github/, root level yamls, etc.
YAML_FILES_TO_CHECK=$(echo "$RAW_FILES_TO_CHECK" | grep -E "^\.buildkite/.*\.ya?ml$" | grep -v "kubernetes/" || true)

# Early exit: If no YAML files were modified, skip validation
if [ -z "$YAML_FILES_TO_CHECK" ]; then
    echo "--- :crossed_fingers: No YAML changes detected. Skipping validation."
    exit 0
fi

# Initialize associative arrays for uniqueness tracking
declare -A PIPELINE_NAMES
declare -A CI_TARGETS

# --- Discover spec directories for uniqueness checks ---
declare -a SPEC_DIRS=("quantization" "parallelism" "models" "features" "rl")
KERNEL_PARENT_DIR="$BUILDKITE_DIR/kernel_microbenchmarks"

echo "--- 📂 Discovering spec directories"
if [[ -d "$KERNEL_PARENT_DIR" ]]; then
    while IFS= read -r dir; do
        # Add subdirectories under kernel_microbenchmarks to SPEC_DIRS
        SPEC_DIRS+=("${dir#"$BUILDKITE_DIR"/}")
    done < <(find "$KERNEL_PARENT_DIR" -maxdepth 1 -mindepth 1 -type d)
fi

# --- Perform Uniqueness Checks for pipeline-name and CI_TARGET in SPEC_DIRS ---
echo "--- 🔍 Checking metadata uniqueness in spec folders"
for folder in "${SPEC_DIRS[@]}"; do
    full_path="$BUILDKITE_DIR/$folder"
    [[ ! -d "$full_path" ]] && continue

    while IFS= read -r -d '' file; do
        # Extract pipeline-name from comment
        P_NAME_LINE=$(awk '/^[[:space:]]*#[[:space:]]*pipeline-name:/ {print $0; exit}' "$file")
        P_NAME=$(echo "${P_NAME_LINE#*:}" | xargs)

        # Extract CI_TARGET value
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
    done < <(find "$full_path" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done

VALIDATE_ARGS=()

echo "--- 📂 Preparing files for validation"

# Iterate through the list to build the arguments array and check file existence
while IFS= read -r file; do
    [ -z "$file" ] && continue

    if [ ! -f "$file" ]; then
        echo "Skipping deleted file: $file"
        continue
    fi

    echo "Adding to validation list: $file"
    VALIDATE_ARGS+=("--file" "$file")

done < <(echo "$YAML_FILES_TO_CHECK")

echo "--- 🔍 Validating changed YAML files"
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
