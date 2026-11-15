#!/usr/bin/env python3
"""Read-only PICO link/button check; no model, UDP sender, ROS or robot commands."""

import argparse
import json
import time

from supermimic.teleop.pico import PicoSource, PicoUpperBodySource


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--input-mode", choices=["full-body", "upper-body"], default="full-body")
    args = p.parse_args()
    if not 0 < args.seconds <= 300:
        p.error("--seconds must be in (0,300]")
    source = PicoUpperBodySource() if args.input_mode == "upper-body" else PicoSource()
    previous = None
    counts = {"tracking_updates": 0, "body_updates": 0, "A_presses": 0, "B_presses": 0}
    start = due = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            if args.input_mode == "upper-body":
                source.read()  # validates actual head/controller poses for the diagnostic
            state = source.status()
            if previous is not None:
                for field, counter in [
                    ("tracking_timestamp_ns", "tracking_updates"),
                    ("body_timestamp_ns", "body_updates"),
                ]:
                    counts[counter] += int(state[field] > 0 and state[field] != previous[field])
                for button in ["A", "B"]:
                    counts[button + "_presses"] += int(state[button] and not previous[button])
            previous = state
            if time.monotonic() >= due:
                print(json.dumps({**state, **counts}, ensure_ascii=False), flush=True)
                print(source.waiting_message(), flush=True)
                due = time.monotonic() + 1.0
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
