import asyncio

from asyncio.tasks import Task

import pytest

from sanic import Sanic, TaskExistsError, text
from sanic.application.state import ApplicationServerInfo, ServerStage


try:
    from unittest.mock import AsyncMock
except ImportError:
    from tests.asyncmock import AsyncMock  # type: ignore

pytestmark = pytest.mark.asyncio


async def runner(steps: int | None = None, interval: float = 0.01):
    count = 0
    while steps is None or count < steps:
        await asyncio.sleep(interval)
        count += 1
    return count


@pytest.fixture(autouse=True)
def mark_app_running(app: Sanic):
    app.state.server_info.append(
        ApplicationServerInfo(
            stage=ServerStage.SERVING, settings={}, server=AsyncMock()
        )
    )


# --------------------------------------------------------------------- #
# Registration: reject, reuse, replace
# --------------------------------------------------------------------- #


async def test_duplicate_name_is_rejected_and_keeps_incumbent(app: Sanic):
    """The original incident: a second job with the same name must not
    silently clobber the first handle."""
    first = app.add_task(runner(), name="compensation")
    second_started = False

    async def second():
        nonlocal second_started
        second_started = True
        await asyncio.sleep(10)

    await asyncio.sleep(0)

    with pytest.raises(TaskExistsError) as excinfo:
        app.add_task(second(), name="compensation")

    # The rejected work never ran, the incumbent is untouched and still
    # resolvable by name.
    assert not second_started
    assert excinfo.value.name == "compensation"
    assert excinfo.value.task is first
    assert app.get_task("compensation") is first
    assert not first.done()

    first.cancel()
    await asyncio.gather(first, return_exceptions=True)


async def test_caller_can_reuse_existing_task(app: Sanic):
    first = app.add_task(runner(), name="compensation")
    await asyncio.sleep(0)

    with pytest.raises(TaskExistsError) as excinfo:
        app.add_task(runner(), name="compensation")

    reused = app.get_task("compensation")
    assert reused is first is excinfo.value.task

    first.cancel()
    await asyncio.gather(first, return_exceptions=True)


async def test_replace_cancels_incumbent_and_installs_replacement(app: Sanic):
    first = app.add_task(runner(), name="compensation")
    await asyncio.sleep(0)

    second = app.add_task(runner(), name="compensation", replace=True)

    assert second is not first
    assert app.get_task("compensation") is second
    assert not second.done()

    await asyncio.gather(first, return_exceptions=True)
    assert first.cancelled()

    second.cancel()
    await asyncio.gather(second, return_exceptions=True)


async def test_replace_task_waits_for_incumbent_cancellation(app: Sanic):
    first = app.add_task(runner(), name="compensation")
    await asyncio.sleep(0)

    second = await app.replace_task("compensation", runner())

    assert isinstance(second, Task)
    assert first.cancelled() and first.done()
    assert app.get_task("compensation") is second

    second.cancel()
    await asyncio.gather(second, return_exceptions=True)


async def test_finished_name_can_be_registered_again(app: Sanic):
    """Once a job fully drains, its name is free for a new compensation run."""
    first = app.add_task(runner(1), name="compensation")
    await first
    app.purge_tasks()

    second = app.add_task(runner(1), name="compensation")
    assert second is not first
    assert app.get_task("compensation") is second
    await second


async def test_failed_name_is_released_after_purge(app: Sanic):
    """A job that dies with an exception must not hold its name hostage."""

    async def boom():
        raise ValueError("ledger write failed")

    first = app.add_task(boom(), name="compensation")
    with pytest.raises(ValueError):
        await first

    app.purge_tasks()
    assert app.get_task("compensation", raise_exception=False) is None

    second = app.add_task(runner(1), name="compensation")
    assert second is not first
    await second


# --------------------------------------------------------------------- #
# Lifecycle isolation: cancel / purge / completion callback
# --------------------------------------------------------------------- #


async def test_purge_never_evicts_replacement_for_finished_holder(
    app: Sanic,
):
    first = app.add_task(runner(1), name="compensation")
    await asyncio.sleep(0)
    second = app.add_task(runner(), name="compensation", replace=True)
    # The displaced holder finishes (cancelled) after the replacement is
    # installed; purging must key off the current incumbent only.
    await asyncio.gather(first, return_exceptions=True)
    app.purge_tasks()
    assert app.get_task("compensation") is second
    assert not second.done()

    second.cancel()
    await asyncio.gather(second, return_exceptions=True)


async def test_cancel_task_only_hits_current_holder(app: Sanic):
    ledger = []

    async def job(tag):
        while True:
            ledger.append(tag)
            await asyncio.sleep(0.01)

    first = app.add_task(job("first"), name="compensation")
    await asyncio.sleep(0.03)
    second = app.add_task(job("second"), name="compensation", replace=True)
    await asyncio.gather(first, return_exceptions=True)

    # Ops cancels by name: the currently registered job stops, and the
    # (already replaced) first job is not touched again or resurrected.
    await app.cancel_task("compensation")
    assert second.cancelled()
    await asyncio.gather(first, second, return_exceptions=True)

    ledger_after = list(ledger)
    await asyncio.sleep(0.05)
    assert ledger == ledger_after
    assert "first" in ledger_after
    assert "second" in ledger_after


async def test_replaced_incumbent_keeps_running_until_cancel_is_observed(
    app: Sanic,
):
    """Replacement requests cancellation but cannot kill a job that
    suppresses CancelledError at the wrong place -- only the live task is
    ever acted upon, the name binding itself is unambiguous."""
    first = app.add_task(runner(100), name="compensation")
    await asyncio.sleep(0)
    second = app.add_task(runner(100), name="compensation", replace=True)

    # Name resolves deterministically to the replacement.
    assert app.get_task("compensation") is second

    await asyncio.gather(first, second, return_exceptions=True)
    assert first.cancelled()
    second.cancel()
    await asyncio.gather(second, return_exceptions=True)


# --------------------------------------------------------------------- #
# Unnamed tasks and register=False stay compatible
# --------------------------------------------------------------------- #


async def test_unnamed_tasks_are_not_registered(app: Sanic):
    task = app.add_task(runner(1))
    assert isinstance(task, Task)
    assert app._task_registry == {}
    assert app.get_task("anything", raise_exception=False) is None
    await task


async def test_register_false_bypasses_registry_and_conflicts(app: Sanic):
    named = app.add_task(runner(), name="compensation")
    shadow = app.add_task(runner(), name="compensation", register=False)
    await asyncio.sleep(0)

    assert shadow is not named
    assert app.get_task("compensation") is named
    for t in (named, shadow):
        t.cancel()
    await asyncio.gather(named, shadow, return_exceptions=True)


# --------------------------------------------------------------------- #
# Shutdown semantics
# --------------------------------------------------------------------- #


async def test_shutdown_cancels_all_named_tasks(app: Sanic):
    first = app.add_task(runner(), name="compensation-1")
    second = app.add_task(runner(), name="compensation-2")
    await asyncio.sleep(0)

    app.shutdown_tasks(timeout=0)
    await asyncio.gather(first, second, return_exceptions=True)

    assert first.cancelled()
    assert second.cancelled()


async def test_server_tasks_tracked_by_identity_not_name(app: Sanic):
    t1 = app._add_server_task(runner())
    t2 = app._add_server_task(runner())
    await asyncio.sleep(0)

    assert t1 in app._server_tasks
    assert t2 in app._server_tasks
    # Internal server tasks must never occupy the named registry, so they
    # can neither shadow user jobs nor be cancelled by user-facing names.
    assert app.get_task("RunServer", raise_exception=False) is None
    assert "RunServer" not in app._task_registry

    t1.cancel()
    t2.cancel()
    await asyncio.gather(t1, t2, return_exceptions=True)
    # Done callback prunes them automatically.
    assert app._server_tasks == set()
