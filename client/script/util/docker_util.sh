#!/bin/sh
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

case "$1" in
    -d)
    HASH=$2
    FROM=$3
    TO=$4
    echo "Copy from ${HASH}:${FROM} to ${TO}"
    docker cp ${HASH}:/workspace/spatialblend-session/${FROM} ${TO}
    ;;

    -r)
    HASH=$2
    RUN_COMMAND=${@:3}
    echo "Run from spatialblend-session workspace in docker image ${HASH} command: ${RUN_COMMAND}"
    docker exec -i ${HASH} bash -c "cd /workspace/spatialblend-session/ && ${RUN_COMMAND}"
    ;;

    *) echo "`basename $0`: usage: [-d directory] | [-r run options (using bash from /workspace/spatialblend-session)]"
    exit 1
    ;;
esac
