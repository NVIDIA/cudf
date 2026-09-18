# Standalone libcudf C++ Application Demonstrating Use of the Pack and Unpack APIs

This C++ example demonstrates the use of libcudf to pack and unpack table data to and
from device and host memory.

The single `pack_unpack_example` binary supports three modes, selected by a CLI argument:

1. `device` (default) — creates a simple cuDF table on the device and then packs and
   unpacks the table on the device. The original and unpacked tables are printed to
   show no change in content.

2. `host` — creates a simple cuDF table on the device and then packs the table into
   pinned host memory. The table is then unpacked from that host buffer for lazy access
   by the device. The original and unpacked tables are printed to show no change in
   content.

3. `host-copy` — creates a simple cuDF table on the device and then packs the table
   into pinned host memory. The packed bytes are then copied to a different host buffer
   (simulating a host-to-host transfer) and later unpacked for lazy access by the
   device. The original and unpacked tables are printed to show no change in content.

## Compile and Execute

```bash
# Configure project
cmake -S . -B build/

# Build
cmake --build build/ --parallel $PARALLEL_LEVEL

# Execute (mode defaults to "device" when omitted)
build/pack_unpack_example device
build/pack_unpack_example host
build/pack_unpack_example host-copy
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
