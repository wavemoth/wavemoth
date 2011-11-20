from cpython cimport PyBytes_FromStringAndSize
cimport numpy as np

cdef write_bin(stream, char *buf, Py_ssize_t size):
    stream.write(PyBytes_FromStringAndSize(buf, size))

def write_int32(stream, np.int32_t i):
    write_bin(stream, <char*>&i, sizeof(i))

def write_int64(stream, np.int64_t i):
    write_bin(stream, <char*>&i, sizeof(i))

def write_array(stream, arr):
    n = stream.write(bytes(arr.data))

def write_aligned_array(stream, arr):
    pad128(stream)
    n = stream.write(bytes(arr.data))

def pad128(stream):
    i = stream.tell()
    m = i % 16
    if m != 0:
        stream.write(b'\0' * (16 - m))

