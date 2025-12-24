/* Syscall stubs required by newlib for bare-metal */

// Stub for closing files (not used)
int _close(int file) { return -1; }

// Stub for seeking files (not used)
int _lseek(int file, int ptr, int dir) { return 0; }

// Stub for reading files (not used)
int _read(int file, char *ptr, int len) { return 0; }

// Implementation for writing to stdout (used by printf)
// Redirects output to the simulator console via putchar
int _write(int file, char *ptr, int len) {
    for (int i = 0; i < len; i++) {
        // putchar is defined in this file
        putchar(ptr[i]); 
    }
    return len;
}
