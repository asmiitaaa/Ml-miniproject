# Checkpoint loading verification

The saved 20,000-step checkpoint is intact. Both outer and nested ZIP CRC checks pass. Direct `torch.load(BytesIO(member), weights_only=True)` succeeds for every tensor archive, whereas the installed Windows PyTorch reader fails on the nested ZipExtFile stream passed by SB3.

`traffic_eval/model_loading.py` scopes a buffered ZIP proxy to SB3's serialization module during loading and restores it immediately. It buffers only `.pth` member streams, retains SB3's normal validation and weights-only loading, changes no model bytes, and does not patch the standard library globally. All 12policy tensors are exactly equal to the original stored tensors. The original SHA256 is 082809d50a56e642a01729d5367e40e77856634acdb41edd1dd8a630668c9c2e. The real-model regression test verifies this on each test run.

Runtime: Stable-Baselines3 2.9.0, PyTorch 2.15.0, CPU inference, one PyTorch thread. The adapter respects the trained 13-feature order, four primary lanes, binary action space and five-second decision bookkeeping. It does not infer corridor mappings.
