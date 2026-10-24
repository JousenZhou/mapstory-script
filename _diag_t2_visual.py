# -*- coding: utf-8 -*-
"""T2（SendMessage 同步投递）可视化确认：三拍截图 + 差分叠加图，肉眼验证 Moonlight 光标移动。"""
import ctypes
import time

import cv2
import numpy as np
import win32api
import win32con
import win32gui

PW_RENDERFULLCONTENT = 2
hwnd = None


def find_moonlight():
    hits = []
    def cb(h, _):
        if "Moonlight" in win32gui.GetWindowText(h) and win32gui.IsWindowVisible(h):
            hits.append(h)
    win32gui.EnumWindows(cb, None)
    return hits[0] if hits else None


def capture(hwnd):
    l, t, r, b = win32gui.GetClientRect(hwnd)
    w, h = r - l, b - t
    hdc = win32gui.GetDC(hwnd)
    mdc = win32gui.CreateCompatibleDC(hdc)
    bmp = win32gui.CreateCompatibleBitmap(hdc, w, h)
    win32gui.SelectObject(mdc, bmp)
    ctypes.windll.user32.PrintWindow(hwnd, mdc, PW_RENDERFULLCONTENT)
    win32gui.SelectObject(mdc, 0)
    from ctypes import wintypes
    class BH(ctypes.Structure):
        _fields_ = [("biSize", wintypes.UINT), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.UINT),
                    ("biSizeImage", wintypes.UINT), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.UINT), ("biClrImportant", wintypes.UINT)]
    bmi = BH()
    bmi.biSize = ctypes.sizeof(BH)
    bmi.biWidth, bmi.biHeight, bmi.biPlanes, bmi.biBitCount, bmi.biCompression = w, -h, 1, 32, 0
    buf = ctypes.create_string_buffer(w * h * 4)
    ctypes.windll.gdi32.GetDIBits(int(hdc), int(bmp), 0, h, buf, ctypes.byref(bmi), 0)
    win32gui.DeleteObject(bmp)
    win32gui.DeleteDC(mdc)
    win32gui.ReleaseDC(hwnd, hdc)
    return np.frombuffer(buf.raw, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()


hwnd = find_moonlight()
assert hwnd
cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
W, H = cr - cl, cb - ct
P1, P2 = (int(W * 0.30), int(H * 0.40)), (int(W * 0.70), int(H * 0.40))

def send_move(x, y):
    ctypes.windll.user32.SendMessageTimeoutW(hwnd, win32con.WM_MOUSEMOVE, 0, win32api.MAKELONG(x, y), 0, 500, None)

orig = win32gui.GetCursorPos()
win32api.SetCursorPos((win32api.GetSystemMetrics(0) - 5, 5))  # 真实光标挪出窗口。
time.sleep(0.5)

send_move(*P1); time.sleep(0.4); c1 = capture(hwnd)
send_move(*P2); time.sleep(0.4); c2 = capture(hwnd)
send_move(*P1); time.sleep(0.4); c3 = capture(hwnd)

win32api.SetCursorPos(orig)
cv2.imwrite("cache/diag_001619/m_t2_p1.png", c1)
cv2.imwrite("cache/diag_001619/m_t2_p2.png", c2)
cv2.imwrite("cache/diag_001619/m_t2_p1b.png", c3)
for name, a, b in (("p1_vs_p2", c1, c2), ("p2_vs_p1b", c2, c3)):
    d = cv2.absdiff(a, b).max(axis=2)
    mask = (d > 25).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    print(name, "changed:", int((mask > 0).sum()),
          ("centroid=(%d,%d)" % (xs.mean(), ys.mean())) if len(xs) else "")
    vis = a.copy()
    vis[mask > 0] = (0, 0, 255)
    cv2.imwrite(f"cache/diag_001619/m_t2_diff_{name}.png", vis)
print("P1=", P1, "P2=", P2, "cursor restored", orig)
