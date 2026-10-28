import importlib.util
import socket
import threading
from pathlib import Path


def test_connect_tunnel_relays_bytes_and_rejects_unlisted_hosts(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "dns_proxy", Path(__file__).parents[1] / "scripts/canfar_dns_proxy.py"
    )
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    upstream, peer = socket.socketpair()
    original_connect = socket.create_connection

    def connect(address, timeout=None):
        if address == ("192.0.2.1", 443):
            return upstream
        return original_connect(address, timeout=timeout)

    monkeypatch.setattr(socket, "create_connection", connect)
    server = proxy.start_dns_proxy({"example.test": "192.0.2.1"})
    try:
        with original_connect(server.server_address, timeout=2) as client:
            client.sendall(b"CONNECT other.test:443 HTTP/1.1\r\n\r\n")
            assert b"403 Forbidden" in client.recv(4096)

        def echo():
            with peer:
                assert peer.recv(4) == b"ping"
                peer.sendall(b"pong")

        worker = threading.Thread(target=echo)
        worker.start()
        with original_connect(server.server_address, timeout=2) as client:
            client.sendall(b"CONNECT example.test:443 HTTP/1.1\r\n\r\n")
            assert b"200 Connection Established" in client.recv(4096)
            client.sendall(b"ping")
            assert client.recv(4) == b"pong"
        worker.join(2)
        assert not worker.is_alive()
    finally:
        server.shutdown()
        server.server_close()
