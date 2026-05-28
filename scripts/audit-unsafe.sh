#!/bin/bash
# Audit unsafe blocks for SAFETY comments
unsafe_count=$(grep -rn 'unsafe {' rust/src/ | grep -v '// SAFETY:' | grep -v 'tests/' | grep -v '_test' | wc -l)
if [ $unsafe_count -gt 0 ]; then
    echo "ERROR: $unsafe_count unsafe blocks missing SAFETY comments"
    grep -rn 'unsafe {' rust/src/ | grep -v '// SAFETY:' | grep -v 'tests/' | grep -v '_test'
    exit 1
fi
echo "OK: All unsafe blocks have SAFETY comments"
