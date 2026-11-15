# Quotienting cuckoo filter versions

Short notes on the lazy 32-bit-prefix quotient table with its one-hot spill
filter. The spill summary remains in the final 64-bit word of each 64-byte
bucket.

- **v2:** Nine-slot layout with 16-bit suffix lanes when the encoded entry
  fits. The ninth slot needed extra scalar handling.
- **v3:** Kept nine slots and added byte-aligned 24-bit suffixes for entries
  up to 56 bits, avoiding a second SIMD load for the ninth prefix.
- **v4:** Fixed the non-EF layout at eight slots for entries up to 48 bits.
  Compared eight 32-bit prefixes and eight 16-bit suffixes together, removing
  the ninth-slot path.
- **v5:** Rounded each entry width up to whole bytes and packed as many as fit
  in the 56 data bytes before the summary. Used AVX512BW byte shuffle and
  compare, followed by a linear bit-mask reduction. It saved memory but was
  slower than v4 in the Hellman run.
- **v6:** Keeps v5's layout and AVX512BW compare. Reduces equal-byte runs by
  combining masks for runs of 2, 4, and 8, then overlaps those masks for the
  selected entry width. This replaces the linear chain of up to seven ANDs
  with at most three dependent ANDs. Hellman was about 1–13% faster than v5,
  depending on thread count and lookup mode, but remained 5–24% slower than v4.
- **v7:** Keeps v5's layout and AVX512BW compare. Adds the start mask to the
  byte equality mask and checks carry bits at entry boundaries. Any boundary
  carry implies at least one complete entry match. On Hellman it was 2–10%
  faster than v6, but remained 2–12% slower than v4.
- **v8:** Keeps v7's carry detector, but makes the AVX512 matcher non-generic
  over entry width. Compile-time shuffle, start, and boundary mask arrays are
  indexed by the table's runtime width. Hellman was 2–9% faster than v7 in
  every mode/thread combination except one-thread loop mode, which was 1% slower.
- **v9:** Moves the selected shuffle, start, and boundary masks into a
  `ByteMasks` member on each quotient table, initialized once from its entry
  width. On Hellman it was 1–13% faster than v8 except 192-thread prefetch2,
  which was about 1% slower.
- **v10:** Removes runtime AVX-512 feature detection. The AVX-512 matcher is
  compiled when both `avx512f` and `avx512bw` are enabled for the target;
  otherwise lookup uses the scalar fallback. Hellman was within about 2% of
  v9 across modes and thread counts, so this change alone did not materially
  improve lookup speed.
- **v11:** Checks the primary bucket's 1-hot spill bit before running the
  primary byte-SIMD match. When spill is possible, computes the secondary hash
  and entry before checking either bucket, allowing that work to overlap with
  the primary SIMD match. On Hellman, prefetch2 improved 5–6% at 1 and 16
  threads and about 1% at 192 threads versus v10. Other modes stayed within
  about 2% of v10.
