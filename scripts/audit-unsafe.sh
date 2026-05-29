#!/bin/bash
# Audit: every `unsafe {` must have a `// SAFETY:` comment within the
# preceding 3 lines of the same file.
#
# Skips test files, bin/ binaries, generated code, and extern "C" blocks.
#
# Usage: bash scripts/audit-unsafe.sh

set -euo pipefail
cd "$(dirname "$0")/.."

errors=0

while IFS=: read -r file line content; do
    # Skip test files, binary entry points, and generated code
    case "$file" in
        *test*) continue ;;
        */bin/*) continue ;;
        *hal_ops_cpu*) continue ;;
    esac

    # Skip extern "C" unsafe blocks (FFI declarations)
    if echo "$content" | grep -q 'extern.*unsafe'; then
        continue
    fi

    # Check for SAFETY comment in the same line or preceding 3 lines
    found=false
    for i in 0 1 2 3; do
        check_line=$((line - i))
        [ "$check_line" -lt 1 ] && continue
        if sed -n "${check_line}p" "$file" | grep -q '// SAFETY:'; then
            found=true
            break
        fi
    done

    if ! $found; then
        echo "MISSING SAFETY: $file:$line"
        errors=$((errors + 1))
    fi
done < <(grep -rn 'unsafe {' --include='*.rs' rust/src/ || true)

if [ "$errors" -eq 0 ]; then
    echo "OK: All unsafe blocks have SAFETY comments"
else
    echo "FAIL: $errors unsafe block(s) missing SAFETY comment"
    exit 1
fi
