# cargo_dep_sha () {
    first_dep=$(cargo tree --depth 1 | head -2 | tail -1 | cut -d' ' -f2);
    index=$(realpath ${CARGO_HOME:-$HOME/.cargo}/registry/src/index*)
    dep=${1:-$first_dep}
    dep_dir=$(cargo tree -p $dep --depth 0 | sed 's/ v/-/')
    grep "sha1" "$index/$dep_dir/.cargo_vcs_info.json"| sed 's/ \+"sha1": "\|"//g'
# }