#!/usr/bin/env bash

SCRIPTS_DIR=$(dirname "$0")
THIRD_PARTY_DIR=$SCRIPTS_DIR/../cmake_build/thirdparty
#API_VERSION=$(go list -m github.com/milvus-io/milvus-proto/go-api/v2 | awk -F' ' '{print $2}')

if [ ! -d "$THIRD_PARTY_DIR/milvus-proto" ]; then
  mkdir -p $THIRD_PARTY_DIR
  pushd $THIRD_PARTY_DIR
  git clone https://github.com/MrPresent-Han/milvus-proto.git
  cd milvus-proto
  # try tagged version first
  COMMIT_ID=df72ba108e12106ffb2c92b9203d30924a7aa29a
  git reset --hard $COMMIT_ID
  popd
fi
