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

# Directories that require strict reporting metadata and dynamic grouping
declare -a SPEC_DIRS=("quantization" "parallelism" "models" "features" "rl")
KERNEL_PARENT_DIR="$BUILDKITE_DIR/kernel_microbenchmarks"

echo "--- 📂 Discovering spec directories"
if [[ -d "$KERNEL_PARENT_DIR" ]]; then
    while IFS= read -r dir; do
        SPEC_DIRS+=("${dir#"$BUILDKITE_DIR"/}")
    done < <(find "$KERNEL_PARENT_DIR" -maxdepth 1 -mindepth 1 -type d)
fi

# --- Global Uniqueness Enforcement ---
# We scan ALL files in spec folders to prevent ID collisions in reporting
echo "--- Checking for global metadata collisions"

declare -A PIPELINE_NAMES
declare -A CI_TARGETS

for folder in "${SPEC_DIRS[@]}"; do
    full_path="$BUILDKITE_DIR/$folder"
    [[ ! -d "$full_path" ]] && continue

    while IFS= read -r -d '' file; do
        # Extract # pipeline-name: (Handles indentation and internal colons)
        P_NAME_LINE=$(awk '/^[[:space:]]*#[[:space:]]*pipeline-name:/ {print $0; exit}' "$file")
        P_NAME=$(echo "${P_NAME_LINE#*:}" | xargs)

        # Extract CI_TARGET
        C_TARGET_RAW=$(grep -E "^[[:space:]]*CI_TARGET:" "$file" | head -1 || true)
        C_TARGET=$(echo "$C_TARGET_RAW" | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\' | xargs)

        # Check for duplicate pipeline-names
        if [[ -n "$P_NAME" ]]; then
            if [[ -n "${PIPELINE_NAMES[$P_NAME]:-}" ]]; then
                echo "+++ ❌ Error: Duplicate '# pipeline-name: $P_NAME' detected!"
                echo "Conflict: $file and ${PIPELINE_NAMES[$P_NAME]}"
                exit 1
            fi
            PIPELINE_NAMES["$P_NAME"]="$file"
        fi

        # Check for duplicate CI_TARGETs
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

# Pre-filter: Only include .yml or .yaml files located within the .buildkite/ directory
# Using '|| true' to prevent the script from exiting if no matches are found
# This automatically ignores .github/, root level yamls, etc.
YAML_FILES_TO_CHECK=$(echo "$RAW_FILES_TO_CHECK" | grep -E "^\.buildkite/.*\.ya?ml$" || true)

# Early exit: If no YAML files were modified, skip validation
if [[ -z "$YAML_FILES_TO_CHECK" ]]; then
    echo "--- :crossed_fingers: No relevant YAML changes detected."
    exit 0
fi

VALIDATE_ARGS=()
echo "--- 🔍 Validating changed pipeline integrity"

while IFS= read -r file; do
    [[ -z "$file" || ! -f "$file" ]] && continue

    IS_PIPELINE=false
    grep -q "^[[:space:]]*steps:" "$file" && IS_PIPELINE=true

    # Determine if the file is in a spec directory for strict rule enforcement
    IS_SPEC=false
    for dir in "${SPEC_DIRS[@]}"; do
        if [[ "$file" == "$BUILDKITE_DIR/$dir/"* || "$file" == "./$BUILDKITE_DIR/$dir/"* ]]; then
            IS_SPEC=true
            break
        fi
    done

    if [[ "$IS_SPEC" == "true" ]]; then
        # Rule: Specs MUST have valid pipeline-name metadata for reporting
        if ! grep -qiE "^[[:space:]]*#[[:space:]]*pipeline-name:[[:space:]]*.+" "$file"; then
            echo "+++ ❌ Error: $file is in a spec folder but is missing a valid '# pipeline-name:' comment."
            exit 1
        fi
        # Rule: Specs MUST be actual pipelines
        if [[ "$IS_PIPELINE" == "false" ]]; then
            echo "+++ ❌ Error: $file is in a spec folder but is missing the 'steps:' root key."
            exit 1
        fi
    fi

    # Only add to Buildkite syntax list if it's actually a pipeline file
    if [[ "$IS_PIPELINE" == "true" ]]; then
        VALIDATE_ARGS+=("--file" "$file")
    else
        echo "--- ℹ️ Skipping Buildkite syntax check for non-pipeline YAML: $file"
    fi
done < <(echo "$YAML_FILES_TO_CHECK")

# --- Final Syntax Validation ---
if [[ ${#VALIDATE_ARGS[@]} -gt 0 ]]; then
    echo "--- 🧪 Running 'bk pipeline validate' for ${#VALIDATE_ARGS[@]} file(s)"
    bk pipeline validate "${VALIDATE_ARGS[@]}"
fi

echo "+++ ✅ All validations successful"
