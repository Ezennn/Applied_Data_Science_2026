import torch

print("torch version:", torch.__version__)
print("torch built with cuda:", torch.backends.cuda.is_built())
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
    print("gpu capability:", "sm_%d%d" % torch.cuda.get_device_capability(0))
    try:
        x = torch.rand(3, 3, device="cuda")
        y = x * 2
        print("cuda tensor op ok:", y.shape)
    except RuntimeError as e:
        print("\nCUDA runtime error while running a simple GPU op:")
        print(str(e))
        if "no kernel image is available" in str(e) or "not compatible with the current PyTorch installation" in str(e):
            print(
                "\nThis usually means your GPU is newer than the CUDA kernels bundled in this PyTorch wheel "
                "(e.g. your GPU is sm_120 but your PyTorch only supports up to sm_90)."
            )
            print("Fix: install a newer PyTorch build that includes your GPU's SM (often the latest stable or nightly).")
else:
    print("GPU not available")
    if not torch.backends.cuda.is_built() or torch.version.cuda is None:
        print("\nYour PyTorch is a CPU-only build (e.g. '+cpu'), so CUDA cannot work even if you have an NVIDIA GPU.")
        print("Install a CUDA build of PyTorch, then re-run this script.")
import torch; print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_arch_list())