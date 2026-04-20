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

# Replicate discovery logic for spec folders used in dynamic uploads
declare -a SPEC_DIRS=("quantization" "parallelism" "models" "features" "rl")
KERNEL_PARENT_DIR="$BUILDKITE_DIR/kernel_microbenchmarks"

if [[ -d "$KERNEL_PARENT_DIR" ]]; then
    while IFS= read -r dir; do
        SPEC_DIRS+=("${dir#"$BUILDKITE_DIR"/}")
    done < <(find "$KERNEL_PARENT_DIR" -maxdepth 1 -mindepth 1 -type d)
fi

# We check ALL files in spec folders to ensure no internal ID collisions
echo "--- Checking for duplicate pipeline-names and CI_TARGETs"

declare -A PIPELINE_NAMES
declare -A CI_TARGETS

for folder in "${SPEC_DIRS[@]}"; do
    full_path="$BUILDKITE_DIR/$folder"
    [[ ! -d "$full_path" ]] && continue

    while IFS= read -r -d '' file; do
        # Extract pipeline-name
        P_NAME_LINE=$(awk '/^[[:space:]]*#[[:space:]]*pipeline-name:/ {print $0; exit}' "$file")
        P_NAME="${P_NAME_LINE#*:}"
        P_NAME="${P_NAME#"${P_NAME%%[![:space:]]*}"}"
        P_NAME="${P_NAME%"${P_NAME##*[![:space:]]}"}"

        # Extract CI_TARGET
        C_TARGET=$(grep -E "^[[:space:]]*CI_TARGET:" "$file" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\' )
        C_TARGET="${C_TARGET%"${C_TARGET##*[![:space:]]}"}"

        # Check for pipeline-name duplicates
        if [[ -n "$P_NAME" ]]; then
            if [[ -n "${PIPELINE_NAMES[$P_NAME]:-}" ]]; then
                echo "+++ ❌ Error: Duplicate '# pipeline-name: $P_NAME' detected!"
                echo "Conflict between: $file and ${PIPELINE_NAMES[$P_NAME]}"
                exit 1
            fi
            PIPELINE_NAMES["$P_NAME"]="$file"
        fi

        # Check for CI_TARGET duplicates
        if [[ -n "$C_TARGET" ]]; then
            if [[ -n "${CI_TARGETS[$C_TARGET]:-}" ]]; then
                echo "+++ ❌ Error: Duplicate 'CI_TARGET: $C_TARGET' detected!"
                echo "Conflict between: $file and ${CI_TARGETS[$C_TARGET]}"
                exit 1
            fi
            CI_TARGETS["$C_TARGET"]="$file"
        fi

    done < <(find "$full_path" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0)
done

# Per-File Validation (Changed Files Only)
YAML_FILES_TO_CHECK=$(echo "$RAW_FILES_TO_CHECK" | grep -E "^\.buildkite/.*\.ya?ml$" || true)

if [ -z "$YAML_FILES_TO_CHECK" ]; then
    echo "--- :crossed_fingers: No pipeline changes detected. Skipping file validation."
    exit 0
fi

VALIDATE_ARGS=()
echo "--- 📂 Validating modified pipeline integrity"

while IFS= read -r file; do
    [ -z "$file" ] && continue
    [ ! -f "$file" ] && continue

    # Check for unreplaced template placeholders
    if grep -qE "\{[A-Z0-9_]+\}" "$file"; then
        echo "+++ ❌ Error: $file contains unreplaced placeholders (e.g., {MODEL_NAME})."
        exit 1
    fi

    # Spec-Specific Metadata Presence Rules
    IS_SPEC=false
    for dir in "${SPEC_DIRS[@]}"; do
        if [[ "$file" == "$BUILDKITE_DIR/$dir/"* ]]; then
            IS_SPEC=true; break
        fi
    done

    if [[ "$IS_SPEC" == "true" ]]; then
        # Rule: Spec files MUST have the comment for upload metadata
        if ! grep -q "^# ?pipeline-name: .." "$file"; then
            echo "+++ ❌ Error: $file is missing the required '# pipeline-name:' comment."
            exit 1
        fi
        
        # Rule: Spec files MUST have 'steps:' for fragment stripping logic
        if ! grep -q "^steps:" "$file"; then
            echo "+++ ❌ Error: $file is missing the 'steps:' root key."
            exit 1
        fi
    fi

    VALIDATE_ARGS+=("--file" "$file")
done < <(echo "$YAML_FILES_TO_CHECK")

echo "--- 🔍 Validating changed YAML files"
if [ ${#VALIDATE_ARGS[@]} -gt 0 ]; then
    if ! bk pipeline validate "${VALIDATE_ARGS[@]}"; then
        echo "+++ ❌ Syntax Validation Failed"
        exit 1
    fi
fi

echo "+++ ✅ All validations successful"
