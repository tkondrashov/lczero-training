#!/usr/bin/env bash

protoc --proto_path=lib/lczero-common --python_out=lib lib/lczero-common/proto/net.proto
protoc --proto_path=lib/lczero-common --python_out=lib lib/lczero-common/proto/chunk.proto
touch lib/proto/__init__.py
