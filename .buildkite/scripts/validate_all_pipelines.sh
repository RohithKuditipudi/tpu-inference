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

# Initialize associative arrays for uniqueness tracking
declare -A PIPELINE_NAMES
declare -A CI_TARGETS

# --- Define specific directories that require strict metadata checks (Spec Folders) ---
declare -a SPEC_DIRS=("quantization" "parallelism" "models" "features" "rl")
KERNEL_PARENT_DIR="$BUILDKITE_DIR/kernel_microbenchmarks"

# Discover subdirectories under kernel_microbenchmarks and add them to SPEC_DIRS
echo "--- 📂 Discovering spec directories"
if [[ -d "$KERNEL_PARENT_DIR" ]]; then
    while IFS= read -r dir; do
        # Strip the .buildkite/ prefix to match the relative path format
        SPEC_DIRS+=("${dir#"$BUILDKITE_DIR"/}")
    done < <(find "$KERNEL_PARENT_DIR" -maxdepth 1 -mindepth 1 -type d)
fi

# --- Perform Uniqueness Checks for pipeline-name and CI_TARGET in SPEC_DIRS ---
echo "--- 🔍 Checking metadata uniqueness in spec folders"
for folder in "${SPEC_DIRS[@]}"; do
    full_path="$BUILDKITE_DIR/$folder"
    
    # Skip if the directory does not exist
    [[ ! -d "$full_path" ]] && continue

    while IFS= read -r -d '' file; do
        # Extract the line containing the pipeline-name comment
        P_NAME_LINE=$(awk '/^[[:space:]]*#[[:space:]]*pipeline-name:/ {print $0; exit}' "$file")
        P_NAME=$(echo "${P_NAME_LINE#*:}" | xargs)

        # Extract the value of the CI_TARGET field
        C_TARGET_RAW=$(grep -E "^[[:space:]]*CI_TARGET:" "$file" | head -1 || true)
        C_TARGET=$(echo "$C_TARGET_RAW" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\' | xargs)

        # Validate pipeline-name uniqueness
        if [[ -n "$P_NAME" ]]; then
            if [[ -n "${PIPELINE_NAMES[$P_NAME]:-}" ]]; then
                echo "+++ ❌ Error: Duplicate '# pipeline-name: $P_NAME' detected!"
                echo "Conflict: $file and ${PIPELINE_NAMES[$P_NAME]}"
                exit 1
            fi
            PIPELINE_NAMES["$P_NAME"]="$file"
        fi

        # Validate CI_TARGET uniqueness
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

# --- Build arguments for Buildkite syntax validation for ALL changed .buildkite/ YAMLs ---
VALIDATE_ARGS=()

echo "--- 📂 Preparing changed files for validation"
while IFS= read -r file; do
    [ -z "$file" ] && continue

    # Ensure the file still exists (handles deleted files in PRs)
    if [ ! -f "$file" ]; then
        echo "Skipping deleted file: $file"
        continue
    fi

    echo "Adding to validation list: $file"
    VALIDATE_ARGS+=("--file" "$file")

done < <(echo "$YAML_FILES_TO_CHECK")

# --- Execute Buildkite Pipeline Validation ---
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
