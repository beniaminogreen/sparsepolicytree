// We need to forward routine registration from C to Rust
// to avoid the linker removing the static library.

void R_init_exhaustivetree_extendr(void *dll);

void R_init_exhaustivetree(void *dll) {
    R_init_exhaustivetree_extendr(dll);
}
