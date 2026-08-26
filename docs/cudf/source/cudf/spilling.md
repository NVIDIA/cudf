(spilling-user-doc)=

# Spilling

cuDF can move eligible buffers from GPU device memory to CPU host memory when spilling is enabled.
This can help workloads whose cuDF buffers exceed available device memory.
Spilling is disabled by default because moving data between host and device memory adds transfer cost.
When cuDF needs a device pointer for spilled data, cuDF moves that data back to device memory automatically.
Spilling can reduce device memory pressure, but it cannot guarantee that every allocation succeeds.

## Enabling Spilling

Configure spilling before creating cuDF objects.
The global spill manager is initialized from the current option values and then reused.

Use {py:func}`cudf.set_option` from Python:

```python
import cudf

cudf.set_option("spill", True)
```

Or set the environment variable before starting Python:

```bash
CUDF_SPILL=on python your_script.py
```

## Configuration

Spilling uses these public options and matching environment variables:

| cuDF option | Environment variable | Default | Effect |
| --- | --- | --- | --- |
| `spill` | `CUDF_SPILL` | `False` | Enables the global spill manager. |
| `spill_on_demand` | `CUDF_SPILL_ON_DEMAND` | `True` | Registers an RMM out-of-memory handler when spilling is enabled. |
| `spill_device_limit` | `CUDF_SPILL_DEVICE_LIMIT` | `None` | Sets a soft byte limit over manager-tracked, currently unspilled cuDF buffers. |
| `spill_stats` | `CUDF_SPILL_STATS` | `0` | Enables duration and byte totals for spill and unspill transfers at level 1 and tracebacks for permanently exposed spillable buffers at level 2 or above. |

`spill_on_demand`, `spill_device_limit`, and `spill_stats` have no effect when `spill` is disabled.
See {ref}`api.options` for the public options reference.

## Behavior and Limits

cuDF spills eligible buffers in least-recently-accessed order.
A spilled buffer returns to device memory when cuDF needs a device pointer for it.
`spill_device_limit` is a soft byte limit over cuDF buffers that the spill manager tracks and that are currently unspilled.
It is not a limit on total GPU memory usage.
Some buffers can be temporarily or permanently unspillable while their device pointers are exposed or in use, and those buffers can cause the tracked unspilled total to exceed the limit.
Spilling can also fail to free enough device memory for an allocation.

See {ref}`Buffer-design` for implementation details.
