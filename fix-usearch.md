# USearch restore() Performance Fix

## Bug

`Index.restore()` has O(n^2) behavior for indexes with sparse/non-sequential keys.
Upstream issue: [unum-cloud/usearch#514](https://github.com/unum-cloud/usearch/issues/514)

### Root Cause

`reindex_keys_()` in `index_dense.hpp` rebuilds a `slot_lookup_` hash set (linear probing) after every load/view.
The hash function in `lookup_key_hash_t` produces pathological collisions for sparse integer keys, causing
linear probing to degenerate. For 3M vectors with sparse keys (IDs spanning 1-55M), this takes 175 seconds.
The same code completes in 0.7s for 10M vectors with sequential keys.

### Evidence

```
enable_key_lookups=True:   3M sparse keys → 175.0s | 10M sequential keys → 0.7s
enable_key_lookups=False:  3M sparse keys →   0.1s | 10M sequential keys → 0.3s
```

## Workaround (applied in corp-entity-db)

Pass `enable_key_lookups=False` to `Index.restore()`. We only use HNSW search (which returns keys from
graph nodes), never key→slot reverse lookup. This drops all index loads to <5s.

## Fork Fixes (for upstream PR)

Fork `unum-cloud/usearch` → `Corp-o-Rate-Community/usearch`, branch `fix/reindex-keys-performance`.

### Fix 1: Replace hash function with splitmix64

**File**: `include/usearch/index.hpp` (or wherever `lookup_key_hash_t` is defined)

The current hash function produces clustering for sparse integer keys. Replace with splitmix64:

```cpp
struct lookup_key_hash_t {
    std::size_t operator()(key_and_slot_t const& ks) const noexcept {
        uint64_t x = static_cast<uint64_t>(ks.key);
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<std::size_t>(x);
    }
};
```

### Fix 2: Pre-reserve hash set in reindex_keys_()

**File**: `include/usearch/index_dense.hpp`, method `reindex_keys_()`

The hash set starts at a small default capacity and grows incrementally during reindexing,
triggering multiple full-table rehashes. Pre-reserve to the known size:

```cpp
void reindex_keys_() {
    std::size_t count_total = typed_->size();
    slot_lookup_.clear();
    slot_lookup_.reserve(count_total);  // Avoid incremental growth + rehash
    // ... existing insertion loop
}
```

### Fix 3: Remove redundant try_reserve in Python binding

**File**: `python/lib.cpp`

Both `load_index_from_path` and `view_index_from_path` call `try_reserve()` after load/view.
But load/view already called `reindex_keys_()` which built the hash set. The second
`slot_lookup_.try_reserve()` is redundant — it may trigger another rehash for no reason.

```cpp
template <typename index_at>
void load_index_from_path(index_at& index, std::string const& path, progress_func_t const& progress) {
    index.load(path.c_str(), {}, progress_t{progress}).error.raise();
    // reindex_keys_() already ran inside load() — only reserve thread contexts, not slot_lookup_
    std::size_t threads = std::thread::hardware_concurrency();
    if (!index.try_reserve(index_limits_t(index.size(), threads)))
        throw std::invalid_argument("Out of memory!");
}
```

The `try_reserve` should skip `slot_lookup_.try_reserve()` if the hash set is already populated
(i.e., `slot_lookup_.size() >= limits.members`).

## Testing

1. Build fork from source: `pip install -e .` from the forked repo
2. Load a 3M-vector index with sparse keys (the hot shard from corp-entity-db)
3. Verify load time drops from 175s to <2s with default `enable_key_lookups=True`
4. Run existing USearch test suite to verify no regressions
