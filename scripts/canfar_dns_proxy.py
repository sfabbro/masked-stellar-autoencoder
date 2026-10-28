"""Loopback HTTPS CONNECT tunnel for CANFAR workers with broken DNS.

TLS stays end to end between the package client and the original hostname.
Only explicitly supplied public hosts on port 443 are accepted.
"""

import os
import select
import socket
import socketserver
import threading


def start_dns_proxy(addresses):
    class Tunnel(socketserver.BaseRequestHandler):
        def handle(self):
            header = b""
            while b"\r\n\r\n" not in header:
                block = self.request.recv(4096)
                if not block:
                    return
                header += block
                if len(header) > 16384:
                    return
            method, authority, _ = header.split(b"\r\n", 1)[0].decode("ascii").split()
            host, port = authority.rsplit(":", 1)
            if method != "CONNECT" or port != "443" or host not in addresses:
                self.request.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            try:
                upstream = socket.create_connection((addresses[host], 443), timeout=30)
            except OSError:
                self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                return
            with upstream:
                self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                sockets = [self.request, upstream]
                while True:
                    readable, _, _ = select.select(sockets, [], [], 120)
                    if not readable:
                        return
                    for current in readable:
                        data = current.recv(65536)
                        if not data:
                            return
                        (upstream if current is self.request else self.request).sendall(
                            data
                        )

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server(("127.0.0.1", 0), Tunnel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proxy = f"http://127.0.0.1:{server.server_address[1]}"
    os.environ.update(
        HTTPS_PROXY=proxy,
        HTTP_PROXY=proxy,
        https_proxy=proxy,
        http_proxy=proxy,
        NO_PROXY="localhost,127.0.0.1",
        no_proxy="localhost,127.0.0.1",
    )
    return server
