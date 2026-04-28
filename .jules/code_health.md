## 2024-05-18 - Refactored complex __init__ in TabResnetWrapper
**What:** Simplified `TabResnetWrapper.__init__` by extracting initialization logic into `_ensure_initialized()`. Added `@property` for `featurescaler` and `opt` to maintain backward compatibility.
**Why:** To resolve linter complexity issues and adhere to `scikit-learn` conventions which expect `__init__` arguments to map directly to identically-named attributes.
**Learning:** `BaseEstimator` instances must not validate or modify constructor arguments in `__init__`. Use lazily-evaluated properties or `_ensure_initialized()` at the entry points to safely bootstrap dependencies without breaking compatibility.
