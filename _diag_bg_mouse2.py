# -*- coding: utf-8 -*-
"""后台鼠标注入实证探测：对游戏窗口逐个测试投递手法，帧差判定游戏画面是否有反应。

纯后台手法（不动真实鼠标、不抢焦点）：
  T0 PostMessage WM_MOUSEMOVE（顶层窗口，客户区坐标）
  T1 AttachThreadInput 伪造激活态 + PostMessage（游戏只在激活态跟踪鼠标时有效）
  T2 SendMessageTimeout 同步投递
  T3 向全部子窗口投递
阳性对照：
  R0 SetCursorPos 真实移动（验证帧差方法可见游戏自绘光标；结束还原原位置）
判定：两次移动点 P1/P2 的截图差分——局部变化≈游戏光标/悬停响应。
"""
import ctypes
import time

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32process

GAME_TITLE = "goodgoodstudydaydayup"
PW_RENDERFULLCONTENT = 2

def find_game():
    hits = []
    def cb(hwnd, _):
        t = win32gui.GetWindowText(hwnd)
        if t and GAME_TITLE in t and win32gui.IsWindowVisible(hwnd):
            l, tp, r, b = win32gui.GetClientRect(hwnd)
            hits.append(((r - l) * (b - tp), hwnd))
    win32gui.EnumWindows(cb, None)
    return max(hits)[1] if hits else None


def capture(hwnd):
    l, t, r, b = win32gui.GetClientRect(hwnd)
    w, h = r - l, b - t
    hdc = win32gui.GetDC(hwnd)
    mdc = win32gui.CreateCompatibleDC(hdc)
    bmp = win32gui.CreateCompatibleBitmap(hdc, w, h)
    win32gui.SelectObject(mdc, bmp)
    result = ctypes.windll.user32.PrintWindow(hwnd, mdc, PW_RENDERFULLCONTENT)
    from ctypes import wintypes
    class BITMAPHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.UINT), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.UINT),
                    ("biSizeImage", wintypes.UINT), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.UINT), ("biClrImportant", wintypes.UINT)]
    win32gui.SelectObject(mdc, 0)  # 取消选择：GetDIBits 要求位图不在 DC 中。
    bmi = BITMAPHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0  # BI_RGB
    buf = ctypes.create_string_buffer(w * h * 4)
    if not ctypes.windll.gdi32.GetDIBits(int(hdc), int(bmp), 0, h, buf, ctypes.byref(bmi), 0):
        win32gui.DeleteObject(bmp)
        win32gui.DeleteDC(mdc)
        win32gui.ReleaseDC(hwnd, hdc)
        return None
    data = buf.raw
    win32gui.DeleteObject(bmp)
    win32gui.DeleteDC(mdc)
    win32gui.ReleaseDC(hwnd, hdc)
    if not result:
        return None
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
    return arr[:, :, :3].copy()


def diff_stats(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    d = cv2.absdiff(a, b).max(axis=2)
    mask = d > 25
    count = int(mask.sum())
    if count == 0:
        return {"count": 0}
    ys, xs = np.nonzero(mask)
    return {"count": count, "cx": int(xs.mean()), "cy": int(ys.mean())}


def children(hwnd):
    out = []
    def cb(h, _):
        out.append(h)
    win32gui.EnumChildWindows(hwnd, cb, None)
    return out


def post_move(hwnd, x, y):
    lparam = win32api.MAKELONG(x, y)
    win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)


def send_move(hwnd, x, y):
    lparam = win32api.MAKELONG(x, y)
    ctypes.windll.user32.SendMessageTimeoutW(hwnd, win32con.WM_MOUSEMOVE, 0, lparam, 0, 500, None)


def activate_forgery(hwnd):
    """AttachThreadInput + SetActiveWindow/SetFocus：让游戏线程认为自己是激活态（不置顶）。"""
    user32 = ctypes.windll.user32
    tid = win32api.GetCurrentThreadId()
    _, target_tid = win32process.GetWindowThreadProcessId(hwnd)
    attached = False
    try:
        attached = user32.AttachThreadInput(tid, target_tid, True)
        user32.SetActiveWindow(hwnd)
        user32.SetFocus(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(tid, target_tid, False)


def probe(name, mover, settle=0.35):
    """在 P1/P2 间移动两次，返回帧差摘要。"""
    shots = []
    for pt in (P1, P2):
        mover(*pt)
        time.sleep(settle)
        shots.append(capture(hwnd))
    d12 = diff_stats(shots[0], shots[1])
    print(f"[{name}] P1->P2 diff: {d12}")
    return d12


hwnd = find_game()
assert hwnd, "game window not found"
if win32gui.IsIconic(hwnd):  # 最小化时画面不刷新，还原后再测。
    print("window minimized, restoring...")
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(1.5)
cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
W, H = cr - cl, cb - ct
P1, P2 = (int(W * 0.25), int(H * 0.5)), (int(W * 0.75), int(H * 0.5))
print(f"hwnd={hwnd} client={W}x{H} P1={P1} P2={P2}")

base = capture(hwnd)
print("capture ok:", base is not None, base.shape if base is not None else "")

# 渲染活性自检：游戏未聚焦时可能暂停渲染（画面冻结），此时一切探测都无意义。
win32gui.SetForegroundWindow(hwnd)
time.sleep(0.8)
_a = capture(hwnd)
time.sleep(0.4)
_b = capture(hwnd)
_alive = diff_stats(_a, _b)
print("render alive (no input):", _alive)
if not _alive or _alive["count"] < 50:
    print("!! 画面未刷新（游戏暂停渲染），探测结果无效。")

kids = children(hwnd)
print("children:", [(k, win32gui.GetClassName(k)) for k in kids])

probe("T0 post-top ", lambda x, y: post_move(hwnd, x, y))
activate_forgery(hwnd)
probe("T1 act+post", lambda x, y: post_move(hwnd, x, y))
activate_forgery(hwnd)
probe("T2 send    ", lambda x, y: send_move(hwnd, x, y))
probe("T3 children", lambda x, y: [post_move(k, x, y) for k in kids] if kids else post_move(hwnd, x, y))
probe("T4 post+setcursor", lambda x, y: (post_move(hwnd, x, y),
                                          win32gui.PostMessage(hwnd, win32con.WM_SETCURSOR, hwnd, win32api.MAKELONG(win32con.HTCLIENT, win32con.WM_MOUSEMOVE))))

# 阳性对照：真实光标移动（结束后还原）。
orig = win32gui.GetCursorPos()
def real_move(x, y):
    sx, sy = win32gui.ClientToScreen(hwnd, (x, y))
    win32api.SetCursorPos((sx, sy))
try:
    probe("R0 real   ", real_move, settle=0.3)
finally:
    win32api.SetCursorPos(orig)
    print("cursor restored to", orig)
