#!/usr/bin/env bash
# CMake caches contain absolute source/build paths and cannot be relocated.
prepare_cmake_build_dir() {
    local source_dir="$1" build_dir="$2"
    local cache="${build_dir}/CMakeCache.txt"
    if [[ -f "$cache" ]]; then
        local cached_source cached_build current_source current_build backup
        cached_source=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache")
        cached_build=$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$cache")
        current_source=$(cd "$source_dir" && pwd -P)
        current_build=$(cd "$build_dir" && pwd -P)
        if [[ ( -n "$cached_source" && "$cached_source" != "$current_source" ) ||
              ( -n "$cached_build" && "$cached_build" != "$current_build" ) ]]; then
            backup="${build_dir}.relocated.$(date +%Y%m%d-%H%M%S).$$"
            while [[ -e "$backup" ]]; do backup="${backup}.1"; done
            echo "检测到复制/移动后的 CMake 缓存，保留旧构建目录：$backup"
            echo "  原源码：$cached_source"
            echo "  当前源码：$current_source"
            mv -- "$build_dir" "$backup"
        fi
    fi
    mkdir -p "$build_dir"
}
