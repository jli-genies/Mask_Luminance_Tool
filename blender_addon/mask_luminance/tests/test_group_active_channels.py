"""group_active_channels() must match run_channel_pipeline's own grouping exactly.

It was factored out of run_channel_pipeline (see that function's history) so
a stepped/modal bake operator can process one work item per tick. These
tests pin down the two behaviors that matter for that use: one item per
ungrouped channel, and blend-group channels collapsing into a single item at
the position of their first occurrence.
"""

from __future__ import annotations

from mask_luminance.core import blend as core_blend


def _channel(name: str, blend_group=None) -> core_blend.MaskChannel:
    return core_blend.MaskChannel(name=name, mask_path="unused", enabled=True, blend_group=blend_group)


def test_ungrouped_channels_are_each_their_own_item():
    channels = [_channel("a"), _channel("b"), _channel("c")]
    items = core_blend.group_active_channels(channels)
    assert items == channels


def test_blend_group_channels_collapse_into_one_item_at_first_occurrence():
    a, hl1, other, hl2 = _channel("a"), _channel("hl1", "hl"), _channel("other"), _channel("hl2", "hl")
    items = core_blend.group_active_channels([a, hl1, other, hl2])

    assert items[0] is a
    assert items[1] == [hl1, hl2]
    assert items[2] is other
    assert len(items) == 3


def test_two_separate_blend_groups_stay_separate():
    a1, a2 = _channel("a1", "groupA"), _channel("a2", "groupA")
    b1, b2 = _channel("b1", "groupB"), _channel("b2", "groupB")
    items = core_blend.group_active_channels([a1, b1, a2, b2])

    assert items == [[a1, a2], [b1, b2]]


def test_empty_input_gives_empty_output():
    assert core_blend.group_active_channels([]) == []
