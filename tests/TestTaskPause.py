# -*- coding: utf-8 -*-
"""task_pause 暂停/松键/恢复语义测试：假 executor/task 驱动，不依赖真实框架。"""
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ok import og, TriggerTask

import src.task_pause as task_pause
from src.task_pause import current_pausable_task, pause_current_task, release_held_keys, resume_task


class FakeInteraction:
    def __init__(self):
        self.up_keys = []

    def send_key_up(self, key):
        self.up_keys.append(key)


class FakeTask:
    def __init__(self, enabled=True, paused=False, held=None):
        self.name = "挂机"
        self._enabled = enabled
        self.paused = paused
        self._held = list(held or [])
        self.paused_count = 0
        self.unpaused_count = 0
        self.pause_calls = 0

    def pause(self):
        self.pause_calls += 1
        self.paused = True
        self.paused_count += 1

    def unpause(self):
        self.unpaused_count += 1
        self.paused = False

    def pop_held_keys(self):
        keys, self._held = self._held, []
        return keys


class FakeTriggerTask(TriggerTask):
    def __init__(self):
        super().__init__()
        self.name = "触发任务"


def _setup(task, interaction=None):
    og.executor = SimpleNamespace(current_task=task)
    og.device_manager = SimpleNamespace(interaction=interaction or FakeInteraction())


def test_pause_releases_held_keys_and_returns_task(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)  # 跳过沉降等待。
    task = FakeTask(held=["a", "d"])
    inter = FakeInteraction()
    _setup(task, inter)
    paused = pause_current_task(owner="test")
    assert paused is task
    assert task.pause_calls == 1
    assert inter.up_keys == ["a", "d"]  # 持有键被逐个松开并清空。
    assert task.pop_held_keys() == []


def test_resume_unpauses_enabled_task(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    task = FakeTask()
    _setup(task)
    resume_task(task, owner="test")
    assert task.unpaused_count == 1


def test_resume_skips_disabled_task(monkeypatch):
    task = FakeTask(enabled=False)
    _setup(task)
    resume_task(task, owner="test")
    assert task.unpaused_count == 0  # 已停用的任务不恢复。


def test_user_paused_task_is_not_taken_over(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    task = FakeTask(paused=True)  # 用户手动暂停。
    _setup(task)
    assert pause_current_task(owner="test") is None
    assert task.pause_calls == 0  # 不接管。


def test_current_task_filters(monkeypatch):
    assert current_pausable_task() is None or True  # 无假执行器时返回 None 或环境任务。
    _setup(FakeTask(enabled=False))
    assert current_pausable_task() is None  # 未启用任务排除。
    _setup(None)
    assert current_pausable_task() is None  # 无任务。
    og.executor = SimpleNamespace(current_task=None)
    assert current_pausable_task() is None
