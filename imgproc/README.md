## Image processing pipelines

### Compilation:
```
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/intel/compilers_and_libraries/linux/bin/intel64/icc -DCMAKE_CXX_FLAGS="-std=gnu++11 -O3" .. && make -j4
```

### Additional paths
```
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin/:$LD_LIBRARY_PATH
```

Best observed time at 1920x1280 images:

| Algorithm | OpenCV (GNU 5.4.0) | TBB (Intel&reg; C++ compiler) | Halide (LLVM 5.0.1) |
|-----------|--------------------|-------------------------------|---------------------|
| rgb2gray  |             0.76ms |                       0.674ms |             0.795ms |
| box_filter|            3.603ms |                       4.779ms |             3.784ms |

CPU: Intel&reg; Core&trade; i5-4460 CPU @ 3.20GHz × 4

OpenCV version: 3.4  
Intel&reg; TBB: 2018_20171205  
Intel&reg; C++ compiler: 18.0.1 20171018
