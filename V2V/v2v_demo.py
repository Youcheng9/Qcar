"""Check the V2V link without driving: run this on both cars (or two terminals).

    python3 v2v_demo.py --id qcar1 [--peer 192.168.1.42] [--brake-after 8]
    python3 v2v_demo.py --id qcar2 [--peer 192.168.1.41]

Each side prints what it hears from the other. --brake-after N makes this
side report "braking" N seconds after start, so you can watch the other side's
decision change from "clear" to "peer-braking".
"""
import argparse
import time

from v2v import V2VLink, limit_throttle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="unique car name")
    parser.add_argument("--peer", help="peer IP (default: broadcast)")
    parser.add_argument("--port", type=int, default=38900)
    parser.add_argument("--brake-after", type=float, default=None)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    start = time.time()
    with V2VLink(args.id, peer_ip=args.peer, port=args.port) as link:
        while time.time() - start < args.duration:
            elapsed = time.time() - start
            braking = (args.brake_after is not None
                       and elapsed >= args.brake_after)
            link.update_state(throttle=0.07, mode="auto", brake=braking)

            peer = link.get_peer()
            throttle, reason = limit_throttle(0.07, peer)
            print("t={:5.1f}s peer={} -> throttle={:.3f} ({})".format(
                elapsed, peer if peer else "none", throttle, reason))
            time.sleep(0.5)


if __name__ == "__main__":
    main()
