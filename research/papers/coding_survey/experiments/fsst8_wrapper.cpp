// Minimal C ABI around the official cwida/fsst library for the Python benchmark.
//
// Archive layout (little endian):
//   uint32 symbol-table bytes, uint64 source bytes, symbol table, FSST payload.

// The compression entry point deliberately creates and exports the symbol table
// on every call.  Its elapsed time is therefore complete warm compression time,
// rather than payload encoding with a pre-trained table.

#include "fsst.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr std::size_t kFrameBytes = sizeof(std::uint32_t) + sizeof(std::uint64_t);

template <typename T>
void store_little_endian(unsigned char* output, T value) {
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output[index] = static_cast<unsigned char>(value >> (8 * index));
    }
}

template <typename T>
T load_little_endian(const unsigned char* input) {
    T value = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<T>(input[index]) << (8 * index);
    }
    return value;
}

}  // namespace

extern "C" std::size_t baseline_fsst8_compress_bound(std::size_t input_bytes) {
    return kFrameBytes + FSST_MAXHEADER + 7 + 2 * input_bytes;
}

extern "C" std::size_t baseline_fsst8_compress(
    const unsigned char* input,
    std::size_t input_bytes,
    unsigned char* output,
    std::size_t output_capacity
) {
    if (input == nullptr || output == nullptr || input_bytes == 0 ||
        output_capacity < baseline_fsst8_compress_bound(input_bytes)) {
        return 0;
    }

    const std::size_t lengths[1] = {input_bytes};
    const unsigned char* strings[1] = {input};
    fsst_encoder_t* encoder = fsst_create(1, lengths, strings, 0);
    if (encoder == nullptr) {
        return 0;
    }

    unsigned char header[FSST_MAXHEADER];
    const std::uint32_t header_bytes = fsst_export(encoder, header);
    const std::size_t payload_offset = kFrameBytes + header_bytes;
    std::size_t compressed_length = 0;
    unsigned char* compressed_string = nullptr;
    const std::size_t compressed_strings = fsst_compress(
        encoder,
        1,
        lengths,
        strings,
        output_capacity - payload_offset,
        output + payload_offset,
        &compressed_length,
        &compressed_string
    );
    fsst_destroy(encoder);
    if (compressed_strings != 1 || compressed_string == nullptr) {
        return 0;
    }

    // fsst_compress may choose a start within the supplied output buffer.
    if (compressed_string != output + payload_offset) {
        std::memmove(output + payload_offset, compressed_string, compressed_length);
    }
    store_little_endian<std::uint32_t>(output, header_bytes);
    store_little_endian<std::uint64_t>(output + sizeof(std::uint32_t), input_bytes);
    std::memcpy(output + kFrameBytes, header, header_bytes);
    return payload_offset + compressed_length;
}

extern "C" std::size_t baseline_fsst8_decompressed_size(
    const unsigned char* archive,
    std::size_t archive_bytes
) {
    if (archive == nullptr || archive_bytes < kFrameBytes) {
        return 0;
    }
    return load_little_endian<std::uint64_t>(archive + sizeof(std::uint32_t));
}

extern "C" std::size_t baseline_fsst8_decompress(
    const unsigned char* archive,
    std::size_t archive_bytes,
    unsigned char* output,
    std::size_t output_capacity
) {
    if (archive == nullptr || output == nullptr || archive_bytes < kFrameBytes) {
        return 0;
    }
    const std::uint32_t header_bytes =
        load_little_endian<std::uint32_t>(archive);
    const std::size_t expected_bytes = baseline_fsst8_decompressed_size(
        archive, archive_bytes
    );
    const std::size_t payload_offset = kFrameBytes + header_bytes;
    if (header_bytes == 0 || header_bytes > FSST_MAXHEADER ||
        payload_offset > archive_bytes || output_capacity < expected_bytes) {
        return 0;
    }

    fsst_decoder_t decoder;
    if (fsst_import(&decoder, archive + kFrameBytes) != header_bytes) {
        return 0;
    }
    const std::size_t decompressed_bytes = fsst_decompress(
        &decoder,
        archive_bytes - payload_offset,
        archive + payload_offset,
        output_capacity,
        output
    );
    return decompressed_bytes == expected_bytes ? decompressed_bytes : 0;
}
