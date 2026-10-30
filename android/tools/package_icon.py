"""Package the supplied transparent artwork into Android launcher resources."""
from pathlib import Path

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "artwork" / "launcher_master.png"
    image = Image.open(source)
    if image.mode != "RGBA" or image.getchannel("A").getextrema() != (0, 255):
        raise ValueError("The launcher master must have genuine transparent and opaque pixels")
    for density, factor in (("mdpi", 1), ("hdpi", 1.5), ("xhdpi", 2), ("xxhdpi", 3), ("xxxhdpi", 4)):
        folder = root / "app" / "src" / "main" / "res" / f"mipmap-{density}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, dp in (("ic_launcher", 48), ("ic_launcher_foreground", 108)):
            size = round(dp * factor)
            image.resize((size, size), Image.Resampling.LANCZOS).save(folder / f"{name}.png")
    image.resize((512, 512), Image.Resampling.LANCZOS).save(root / "artwork" / "launcher_512.png")
    print("Packaged 10 full-colour RGBA launcher resources and a 512 px export.")


if __name__ == "__main__":
    main()
