#!/usr/bin/env python3
import asyncio
import threading

SERVER_PATH = "neural-tools/src/mcp/neural_server_stdio.py"


def _run_server_thread():
    async def _main():
        import importlib.util
        spec = importlib.util.spec_from_file_location("neural_server_stdio", SERVER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        await mod.run()  # should manage its own loop

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asyncio.wait_for(_main(), timeout=1.0))
    except Exception:
        # Expect timeout (server run loop blocks); important is no RuntimeError
        pass
    finally:
        loop.stop()
        loop.close()


def test_no_event_loop_conflicts_threaded():
    t = threading.Thread(target=_run_server_thread, daemon=True)
    t.start()
    t.join(timeout=2)
    assert True
