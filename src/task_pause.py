# -*- coding: utf-8 -*-
"""脚本任务暂停/恢复助手：测谎服务与掉线重登监控共用的联动语义。

语义（沿用实战验证过的口径）：
- 只暂停当前一次性脚本任务（排除 TriggerTask 与未启用任务）；
- 用户手动暂停的任务不接管、解题后也不自动恢复（尊重用户意图）；
- 暂停后等任务线程阻塞到 sleep，再释放其持有的方向/攻击键，避免按键残留；
- 恢复时仅当任务仍处于启用状态才 unpause。
"""
import time

from ok import Logger, TriggerTask, og

logger = Logger.get_logger(__name__)

PAUSE_SETTLE_SECONDS = 0.2  # 暂停后等任务线程阻塞到 sleep 的秒数，再松键。


def current_pausable_task():
    """取当前可暂停的脚本任务（排除触发任务与未启用任务），无则返回 None。"""
    executor = getattr(og, "executor", None)
    if executor is None:
        return None
    task = getattr(executor, "current_task", None)
    if task is None:
        return None
    if isinstance(task, TriggerTask):  # 触发任务的暂停语义不同（全局暂停），不接管。
        return None
    if not getattr(task, "_enabled", False):
        return None
    return task


def release_held_keys(task):
    """释放任务当前持有的按键：任务上报持有键，这里负责实际松键 IO。"""
    device_manager = getattr(og, "device_manager", None)
    interaction = getattr(device_manager, "interaction", None) if device_manager is not None else None
    if interaction is None:
        return
    pop = getattr(task, "pop_held_keys", None)
    if not callable(pop):
        return
    try:
        keys = pop() or []
    except Exception:
        keys = []
    for key in keys:
        try:
            interaction.send_key_up(key)
        except Exception as e:
            logger.warning(f"Release key {key} failed: {e}. 松开按键失败：{key}。")


def pause_current_task(owner="lie detector"):
    """暂停当前脚本任务并释放其持有键，返回被暂停的任务（无需恢复时返回 None）。"""
    task = current_pausable_task()
    if task is None:
        return None
    if getattr(task, "paused", False):  # 用户手动暂停：不接管也不恢复。
        return None
    try:
        task.pause()  # 置 _paused，任务线程在下一次 sleep 处阻塞。
    except Exception as e:
        logger.warning(f"Pause task for {owner} failed: {e}. 暂停脚本任务失败，将独立执行。")
        return None
    time.sleep(PAUSE_SETTLE_SECONDS)  # 等任务线程阻塞到 sleep，确保不再发按键。
    release_held_keys(task)
    logger.info(f"Task paused for {owner}: {getattr(task, 'name', task)}. 已暂停脚本任务。")
    return task


def resume_task(task, owner="lie detector"):
    """恢复被暂停的脚本任务（仅当任务仍处于启用状态）。"""
    if task is None:
        return
    try:
        if getattr(task, "_enabled", False):
            task.unpause()
            logger.info(f"Task resumed after {owner}: {getattr(task, 'name', task)}. 已恢复脚本任务。")
    except Exception as e:
        logger.warning(f"Resume task for {owner} failed: {e}. 恢复脚本任务失败。")
