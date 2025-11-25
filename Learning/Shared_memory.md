
---

# CUDA Shared Memory Cheatsheet

## 1. Why Shared Memory Exists

Global memory is slow. If an algorithm keeps touching the same data, it makes no sense to reload it from global memory every time. Shared memory sits on-chip, much faster, and lets a block reuse data without paying global memory costs.

## 2. Tiling

A **tile** is a chunk of data that a block loads together. Every thread grabs a piece, the block stores it in shared memory, and the threads operate on it many times instead of going back to global memory.

## 3. Halos

Image kernels usually need neighbors. A tile by itself won’t cover edges, so you add a **halo**—extra pixels around the tile—to let each thread access its surrounding region without making extra global reads.

## 4. Shared Memory Banks

Shared memory is split into **32 banks**.
A bank serves one 32-bit word per cycle.
When two threads in a warp hit the same bank, the access becomes serialized.

Bank index formula:

```
bank = (address_in_bytes / 4) % 32
```

## 5. Bank Conflicts

If your row width (pitch) is a multiple of 32, accessing down a column is a recipe for collisions.

Example of a bad layout:

```cpp
__shared__ float tile[32][32];
```

Pitch = 32 →
32 % 32 = 0 →
Every row starts at the same bank →
Column accesses collide.

## 6. Padding Fix (Read This Twice)

A 32×32 tile means each row is 128 bytes.
128 bytes / 4 = 32 words.
32 % 32 = 0 → every row begins on bank 0.

So if thread 0 touches tile[r][0] across multiple rows, they all map to bank 0.
That leads to conflicts when warps read vertically.

The fix is stupidly simple:

```cpp
__shared__ float tile[32][33];
```

Now each row is 33 floats:

* 33 × 4 = 132 bytes
* 132 / 4 = 33
* 33 % 32 = 1

Each row starts one bank later than the last:

* Row 0 → bank 0
* Row 1 → bank 1
* Row 2 → bank 2
* …
* Row 31 → bank 31

Column access no longer collapses onto the same bank.
You’ve staggered the rows, and the conflicts disappear.

**Mental model**
Think of 32 doors. Each row originally started at door 0.
Add padding, and now row N starts at door N.
Everyone walks through different doors. No jam.

**TLDR you won’t forget**

* Shared memory has 32 banks.
* Never make your row width a multiple of 32.
* Add one extra column.
* Conflicts vanish.
* Cost: tiny.

## 7. Coalescing

When 32 threads read consecutive global memory addresses, the hardware can fetch all of it in one transaction. This is **coalesced access** and is crucial for performance.

## 8. Final Summary

* Shared memory stops repeated global loads.
* Tiling boosts locality.
* Halos handle neighborhood operations.
* Padding avoids bank conflicts.
* Coalescing keeps global memory fast.

---

