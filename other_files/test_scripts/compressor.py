import time
import gzip

def compress(input_fd, output_fd):
    """
    Compress the input file descriptor and write to the output file descriptor.
    This is a placeholder function that does not perform any real compression.

    Args:
        input_fd: File descriptor for the input file (The original file).
        output_fd: File descriptor for the output file (Compression file).
    
    Returns:
        float: Time taken to compress the file in seconds.
    """
    start_time = time.time()
    with gzip.GzipFile(fileobj=output_fd, mode='wb') as gz_out:
        gz_out.write(input_fd.read())
    return time.time() - start_time

def decompress(input_fd, output_fd):
    """
    Decompress the input file descriptor and write to the output file descriptor.
    This is a placeholder function that does not perform any real decompression.

    Args:
        input_fd: File descriptor for the input file (Compressed file).
        output_fd: File descriptor for the output file (Decompressed file),
                   this output file need to be same as the original file.
    
    Returns:
        None
    """
    start_time = time.time()
    with gzip.GzipFile(fileobj=input_fd, mode='rb') as gz_in:
        output_fd.write(gz_in.read())
    return time.time() - start_time
