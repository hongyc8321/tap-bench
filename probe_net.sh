#!/usr/bin/env bash
echo "== local v4 egress =="
curl -4 -s -m 10 ifconfig.me || echo V4_FAIL
echo
echo "== probe server ports =="
curl -4 -s -m 8 -o /dev/null -w "port8888: %{http_code}\n" http://36.103.236.54:8888
curl -4 -s -m 8 -o /dev/null -w "port6606: %{http_code}\n" http://36.103.236.54:6606
curl -4 -s -m 8 -o /dev/null -w "port22-banner: %{http_code}\n" telnet://36.103.236.54:22 2>/dev/null || true
echo "== ssh verbose (kex stage) =="
ssh -4 -i /home/rhap/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=12 -p 22 ubuntu@36.103.236.54 echo OK 2>&1 | tail -2
