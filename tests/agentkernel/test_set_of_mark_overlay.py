from __future__ import annotations

from PIL import Image

from northcua.agentkernel.overlay.set_of_mark import (
    SetOfMarkBox,
    SetOfMarkOverlayMark,
    SetOfMarkOverlayRenderer,
    SetOfMarkOverlaySpec,
)


def test_set_of_mark_renderer_keeps_label_for_full_viewport_target(tmp_path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "overlay.png"
    Image.new("RGB", (200, 120), "white").save(source, format="PNG")

    mark = SetOfMarkOverlayMark(
        ref="@e1",
        label="1",
        role="button",
        name="Full viewport target",
        bbox_css=SetOfMarkBox(x=0, y=0, width=200, height=120),
        bbox_image=SetOfMarkBox(x=0, y=0, width=200, height=120),
    )

    artifact = SetOfMarkOverlayRenderer().render(
        source_image_path=str(source),
        output_path=str(output),
        spec=SetOfMarkOverlaySpec(marks=[mark]),
    )

    assert artifact.output_path == str(output)
    assert output.exists()
    assert SetOfMarkOverlayRenderer()._choose_label_rect(
        box_rect=(0, 0, 199, 119),
        chip_size=(18, 16),
        image_size=(200, 120),
        occupied_labels=[],
        target_rects=[(0, 0, 199, 119)],
    )


def test_set_of_mark_renderer_places_small_target_label_outside_target() -> None:
    renderer = SetOfMarkOverlayRenderer()
    target = (50, 50, 150, 64)

    rect = renderer._choose_label_rect(
        box_rect=target,
        chip_size=(18, 16),
        image_size=(240, 160),
        occupied_labels=[],
        target_rects=[target],
    )

    assert rect is not None
    assert renderer._rect_intersection_area(rect, target) == 0
