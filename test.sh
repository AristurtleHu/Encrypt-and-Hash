#!/bin/bash

# Number of iterations
iterations=22

# Initialize total durations (in nanoseconds)
total_duration_ns_test0=0
total_duration_ns_test2=0

echo "Starting $iterations iterations of tests..."
echo "Each iteration consists of: make all && ./program ./testcases/test_0.meta && ./program ./testcases/test_2.meta"

# 1. Run make all
# Redirect stdout and stderr to /dev/null to keep output clean
make all > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "\nError: 'make all' failed on iteration $i. Aborting."
    exit 1
fi

# Loop for the specified number of iterations
for i in $(seq 1 $iterations); do
    # Print progress, \r carriage return to update line in place
    echo -ne "Iteration $i/$iterations\r"

    # 2. Run and time ./program ./testcases/test_0.meta
    start_ns_test0=$(date +%s%N)
    ./program ./testcases/test_0.meta > /dev/null 2>&1
    end_ns_test0=$(date +%s%N)
    duration_ns_test0=$((end_ns_test0 - start_ns_test0))
    total_duration_ns_test0=$((total_duration_ns_test0 + duration_ns_test0))

    # 3. Run and time ./program ./testcases/test_2.meta
    # start_ns_test2=$(date +%s%N)
    # ./program ./testcases/test_2.meta > /dev/null 2>&1
    # end_ns_test2=$(date +%s%N)
    # duration_ns_test2=$((end_ns_test2 - start_ns_test2))
    # total_duration_ns_test2=$((total_duration_ns_test2 + duration_ns_test2))
done

# Ensure the progress line is overwritten by the next output
echo -e "\nIterations complete."

avg_duration_ns_test0=$((total_duration_ns_test0 / iterations))
avg_duration_ns_test2=$((total_duration_ns_test2 / iterations))

echo "-----------------------------------------------------"
echo "Average Execution Times (over $iterations iterations):"
echo "-----------------------------------------------------"
echo "Testcase ./testcases/test_0.meta: $avg_duration_ns_test0 ns"
echo "Testcase ./testcases/test_2.meta: $avg_duration_ns_test2 ns"

avg_duration_ms_test0=$(echo "scale=6; $avg_duration_ns_test0 / 1000000000" | bc)
avg_duration_ms_test2=$(echo "scale=6; $avg_duration_ns_test2 / 1000000000" | bc)
echo ""
echo "In milliseconds (requires 'bc' command):"
echo "Testcase ./testcases/test_0.meta: $avg_duration_ms_test0 s"
echo "Testcase ./testcases/test_2.meta: $avg_duration_ms_test2 s"
echo "-----------------------------------------------------"

exit 0