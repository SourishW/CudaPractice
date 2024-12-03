if [ -z "$1" ]; then
    echo "Usage: $0 <cuda_file>"
    exit 1
fi

nvcc "$1" -o output_executable
if [ $? -eq 0 ]; then
    ./output_executable
else
    echo "Compilation failed."
    exit 1
fi
