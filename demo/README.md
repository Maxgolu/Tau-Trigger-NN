# Final pipeline demo

From the repository root, after installing `requirements.txt`, run:

```bash
python demo/run_demo.py
```

Verify that its machine-readable output still matches the committed contract:

```bash
python demo/verify_demo.py
```

The demo uses deterministic synthetic inputs, so it does not require the private detector dataset or trained checkpoints. It verifies the public final configuration's all-finite pair construction, 4,357-parameter network, maximum pair-to-event aggregation, no-pair behavior, and event-level neural-or-pT decision. It is a functional code demonstration, not a reproduction of the reported physics result; reproducing that number requires the externally supplied dataset and frozen checkpoints described in the result manifest.
