"""V2V: car-to-car state sharing over UDP.

Each car runs one V2VLink. The link broadcasts this car's state at a fixed
rate from a background thread and keeps the latest state of every other car
it hears from. It uses only the standard library, so it can be tested on a
laptop without a QCar.

Packets are small JSON objects:
    {"v": 1, "car_id": "qcar1", "seq": 42, "t": 1712345678.9,
     "throttle": 0.07, "steering": -0.1, "mode": "auto",
     "brake": false, "turn_signal": "left", "obstacle": false}

Freshness is judged by the LOCAL receive time, not the sender's "t", because
the two cars' clocks are not synchronised. "seq" is used to drop duplicated
or out-of-order packets, since UDP guarantees neither.
"""
import json
import socket
import threading
import time

V2V_PORT = 38900  # TestDriveServer.py already uses 38822
PROTOCOL_VERSION = 1
SEND_HZ = 10.0
STALE_AFTER = 0.5  # seconds without a packet before a peer is ignored


class PeerState:
    """Latest state received from one peer car."""

    def __init__(self, packet, received_at):
        self.car_id = packet["car_id"]
        self.seq = packet["seq"]
        self.throttle = float(packet.get("throttle", 0.0))
        self.steering = float(packet.get("steering", 0.0))
        self.mode = packet.get("mode", "manual")
        self.brake = bool(packet.get("brake", False))
        self.turn_signal = packet.get("turn_signal", "none")
        self.obstacle = bool(packet.get("obstacle", False))
        self.received_at = received_at

    def age(self):
        return time.time() - self.received_at

    def __repr__(self):
        return (
            "PeerState({} seq={} throttle={:+.3f} steering={:+.3f} "
            "mode={} brake={} signal={} obstacle={} age={:.2f}s)".format(
                self.car_id, self.seq, self.throttle, self.steering,
                self.mode, self.brake, self.turn_signal, self.obstacle,
                self.age(),
            )
        )


class V2VLink:
    """Broadcasts our state and tracks the other cars' states.

    car_id:   unique name for this car, e.g. "qcar1".
    peer_ip:  send directly to this IP. If None, broadcast to the subnet.
    port:     UDP port we listen on (also the one we send to by default).
    peer_port: send to a different port; only needed to test two links on
              one machine.
    """

    def __init__(self, car_id, peer_ip=None, port=V2V_PORT,
                 send_hz=SEND_HZ, stale_after=STALE_AFTER, peer_port=None):
        self.car_id = car_id
        self.port = port
        self.send_period = 1.0 / send_hz
        self.stale_after = stale_after
        self._dest = (peer_ip or "255.255.255.255", peer_port or port)

        self._state = {
            "throttle": 0.0, "steering": 0.0, "mode": "manual",
            "brake": False, "turn_signal": "none", "obstacle": False,
        }
        self._state_lock = threading.Lock()

        self._peers = {}
        self._peers_lock = threading.Lock()

        self._seq = 0
        self._stop = threading.Event()
        self._threads = []

        # One socket for sending, one for receiving.
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tx.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            # Lets two test processes share the port on one machine.
            self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._rx.bind(("0.0.0.0", port))
        self._rx.settimeout(0.2)  # so the thread can notice stop()

    # ---- lifecycle -------------------------------------------------------

    def start(self):
        for target in (self._send_loop, self._recv_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def stop(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._tx.close()
        self._rx.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- our state -------------------------------------------------------

    def update_state(self, **fields):
        """Set any of: throttle, steering, mode, brake, turn_signal, obstacle."""
        with self._state_lock:
            for key, value in fields.items():
                if key not in self._state:
                    raise KeyError("unknown V2V field: " + key)
                self._state[key] = value

    # ---- peers' state ----------------------------------------------------

    def get_peers(self):
        """Fresh peers only (heard from within stale_after seconds)."""
        with self._peers_lock:
            return [p for p in self._peers.values()
                    if p.age() <= self.stale_after]

    def get_peer(self, car_id=None):
        """One fresh peer, or None. Without car_id, the most recent peer."""
        peers = self.get_peers()
        if car_id is not None:
            peers = [p for p in peers if p.car_id == car_id]
        if not peers:
            return None
        return max(peers, key=lambda p: p.received_at)

    def ever_heard_peer(self):
        with self._peers_lock:
            return bool(self._peers)

    # ---- threads ---------------------------------------------------------

    def _send_loop(self):
        while not self._stop.is_set():
            with self._state_lock:
                packet = dict(self._state)
            self._seq += 1
            packet.update(v=PROTOCOL_VERSION, car_id=self.car_id,
                          seq=self._seq, t=time.time())
            try:
                self._tx.sendto(json.dumps(packet).encode("utf-8"), self._dest)
            except OSError as error:
                # Wi-Fi hiccup: keep going, the peer will treat us as stale.
                print("V2V send failed:", error)
            self._stop.wait(self.send_period)

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                data, _addr = self._rx.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
                if packet["v"] != PROTOCOL_VERSION:
                    continue
                if packet["car_id"] == self.car_id:
                    continue  # our own broadcast
                state = PeerState(packet, time.time())
            except (ValueError, KeyError, TypeError):
                continue  # malformed packet: ignore, never crash the car
            with self._peers_lock:
                old = self._peers.get(state.car_id)
                if old is None or state.seq > old.seq or state.seq < old.seq - 100:
                    # (seq far below old means the peer restarted)
                    self._peers[state.car_id] = state


# ---- driving policy ------------------------------------------------------

def limit_throttle(own_throttle, peer, slow_factor=0.5):
    """Reduce our throttle based on a peer's state. Returns (throttle, reason).

    - peer is braking -> stop,
    - peer reports an obstacle -> slow to slow_factor of our throttle,
    - no fresh peer -> unchanged (the link is advisory, never required).
    """
    if peer is None:
        return own_throttle, "no-peer"
    if peer.brake:
        return 0.0, "peer-braking"
    if peer.obstacle:
        return own_throttle * slow_factor, "peer-obstacle"
    return own_throttle, "clear"
