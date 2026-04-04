#!/usr/bin/env bash
# strategies/generate_proto.sh
set -euo pipefail
PROTO_DIR="../engine/crates/proto/proto"
OUT_DIR="./server/generated"
mkdir -p "$OUT_DIR"
python -m grpc_tools.protoc \
  --proto_path="$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  strategy.proto
# Fix relative imports in generated gRPC stubs
# grpc_tools generates "import strategy_pb2" but we need "from . import strategy_pb2"
sed -i '' 's/^import strategy_pb2 as/from . import strategy_pb2 as/' "$OUT_DIR/strategy_pb2_grpc.py"
touch "$OUT_DIR/__init__.py"
echo "Proto generation complete."
