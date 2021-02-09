"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@llvm-bazel//:terminfo.bzl", "llvm_terminfo_disable")
load("@llvm-bazel//:zlib.bzl", "llvm_zlib_disable")
load("@llvm-bazel//:configure.bzl", "llvm_configure")

def setup_llvm():
    # Disable terminfo and zlib that are bundled with LLVM
    llvm_terminfo_disable(
        name = "llvm_terminfo",
    )
    llvm_zlib_disable(
        name = "llvm_zlib",
    )

    # Now that all LLVM settings are loaded, configure the LLVM project.
    # This mixes llvm-bazel and llvm-project-raw to create llvm-project.
    # The result has both the source code and the BUILD files in one place.
    llvm_configure(
        name = "llvm-project",
        src_path = ".",
        src_workspace = "@llvm-project-raw//:WORKSPACE",
    )
