set pagination off

python
import gdb
import os
import threading
import time


def request_interrupt():
    time.sleep(int(os.environ["CUDF_IWYU_GDB_TIMEOUT_SECONDS"]))
    # Schedule this on GDB's event loop so `run` only returns after the inferior stops.
    gdb.post_event(lambda: gdb.execute("interrupt"))


threading.Thread(target=request_interrupt, daemon=True).start()
end

run
thread apply all bt
quit
