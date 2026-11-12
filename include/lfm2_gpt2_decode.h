#ifndef LFM2_GPT2_DECODE_H
#define LFM2_GPT2_DECODE_H

#include <stddef.h>
#include <stdint.h>

/**
 * Decodes a GPT-2 byte-level UTF-8 vocabulary token into raw UTF-8/byte output.
 *
 * GPT-2 BPE encodes raw bytes in [33..126, 161..172, 174..255] directly as their
 * matching Unicode codepoints (meaning bytes 161..172 and 174..255 are stored as
 * 2-byte UTF-8 sequences U+00A1..U+00FF) and shifts [0..32, 127..160, 173] into
 * U+0100..U+0143 (256..323). This reverses both ranges back into single raw bytes.
 */
static inline size_t lfm2_decode_gpt2_bytes(const char *src_utf8, char *dst, size_t dst_cap) {
    if (!src_utf8 || !dst || dst_cap == 0) return 0;

    const uint8_t *s = (const uint8_t *)src_utf8;
    size_t pos = 0;

    while (*s && pos + 1 < dst_cap) {
        if ((s[0] & 0xE0) == 0xC0 && s[1] != '\0' && (s[1] & 0xC0) == 0x80) {
            uint16_t cp = (uint16_t)(((uint16_t)(s[0] & 0x1F) << 6) | (uint16_t)(s[1] & 0x3F));
            if (cp >= 256 && cp <= 323) {
                uint16_t off = cp - 256;
                uint8_t raw = (off < 33) ? (uint8_t)off :
                              (off < 67) ? (uint8_t)(off - 33 + 127) : 173;
                dst[pos++] = (char)raw;
                s += 2;
                continue;
            }
            if ((cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
                dst[pos++] = (char)(uint8_t)cp;
                s += 2;
                continue;
            }
        }
        dst[pos++] = (char)(*s++);
    }
    dst[pos] = '\0';
    return pos;
}

#endif /* LFM2_GPT2_DECODE_H */

