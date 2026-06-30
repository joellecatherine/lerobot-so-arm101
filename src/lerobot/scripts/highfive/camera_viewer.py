#!/usr/bin/env python
"""Live camera viewer — watches /tmp/highfive_cameras.png and displays it.

Run in a separate terminal while training:
    python -m lerobot.scripts.highfive.camera_viewer
"""
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

IMG_PATH = Path("/tmp/highfive_cameras.png")
REFRESH_MS = 100  # refresh every 100ms


def main():
    root = tk.Tk()
    root.title("Camera feeds (birdseye | wrist)")
    label = tk.Label(root)
    label.pack()

    last_mtime = 0.0

    def update():
        nonlocal last_mtime
        try:
            if IMG_PATH.exists():
                mtime = IMG_PATH.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    img = Image.open(IMG_PATH)
                    photo = ImageTk.PhotoImage(img)
                    label.configure(image=photo)
                    label.image = photo  # prevent GC
        except Exception:
            pass  # file might be mid-write
        root.after(REFRESH_MS, update)

    update()
    root.mainloop()


if __name__ == "__main__":
    main()
