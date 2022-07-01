use libc::{c_void, dlclose, dlopen, dlsym, RTLD_NOW};
use std::ffi::CString;

fn main() {
    println!("Trying to load a library...");
    unsafe {
        // Load the library
        let filename = CString::new("./libhello.so").unwrap();
        let handle = dlopen(filename.as_ptr(), RTLD_NOW);
        if handle.is_null() {
            panic!("Failed to resolve dlopen")
        }

        // Look for the function in the library
        let fun_name = CString::new("hello").unwrap();
        let fun = dlsym(handle, fun_name.as_ptr());
        if fun.is_null() {
            panic!("Failed to resolve '{}'", &fun_name.to_str().unwrap());
        }

        // dlsym returns a C 'void*', cast it to a function pointer
        let fun = std::mem::transmute::<*mut c_void, fn()>(fun);
        fun();

        // Cleanup
        let ret = dlclose(handle);
        if ret != 0 {
            panic!("Error while closing lib");
        }
    }
}