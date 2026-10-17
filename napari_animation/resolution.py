"""Keep authored resolution choices consistent across an animation."""

from dataclasses import replace

from .layer_registry import OBJECT_ID_METADATA


class ResolutionSettings:
    def __init__(self, animation):
        self.animation = animation
        self._listeners = {}
        self.apply_to_all_layers = True
        self._propagating = False
        self._checkboxes = {}
        layers = animation.viewer.layers
        layers.events.inserted.connect(self._inserted)
        layers.events.removed.connect(self._removed)
        for layer in layers:
            self._watch(layer)

    def _inserted(self, event):
        self._watch(event.value)
        self._attach_layer_control(event.value)

    def _removed(self, event):
        self._unwatch(event.value)
        self._checkboxes.pop(event.value, None)

    def _watch(self, layer):
        if (
            layer in self._listeners
            or not getattr(layer, "multiscale", False)
            or layer.metadata.get(OBJECT_ID_METADATA)
            or not hasattr(layer, "locked_data_level")
        ):
            return
        # Preserve an explicit choice made before opening the animation dock.
        if layer.locked_data_level is None:
            layer.locked_data_level = 0

        def changed(event=None):
            animation = self.animation
            if (
                animation._closed
                or animation._navigating
                or animation.renderer.applying
            ):
                return
            if self.apply_to_all_layers and not self._propagating:
                self._propagating = True
                try:
                    level = layer.locked_data_level
                    for other in tuple(self._listeners):
                        if other is layer:
                            continue
                        # A layer with fewer pyramid levels uses its coarsest
                        # available level when the requested one is absent.
                        target = (
                            None
                            if level is None
                            else min(level, len(other.data) - 1)
                        )
                        if other.locked_data_level != target:
                            other.locked_data_level = target
                finally:
                    self._propagating = False
            self._rewrite_layer(layer)

        layer.events.locked_data_level.connect(changed)
        self._listeners[layer] = changed

    def attach_controls(self):
        """Add the shared setting next to napari's resolution dropdowns."""
        for layer in self.animation.viewer.layers:
            self._attach_layer_control(layer)

    def _attach_layer_control(self, layer):
        if layer not in self._listeners or layer in self._checkboxes:
            return
        viewer = self.animation.viewer
        window = getattr(viewer, "window", None)
        qt_viewer = getattr(window, "_qt_viewer", None)
        controls_container = getattr(qt_viewer, "_controls", None)
        if controls_container is None:
            return
        controls = controls_container.widgets.get(layer)
        level_control = getattr(controls, "_multiscale_level_control", None)
        if level_control is None:
            return

        from qtpy.QtWidgets import QCheckBox

        checkbox = QCheckBox("Apply to all layers", controls)
        checkbox.setChecked(self.apply_to_all_layers)
        checkbox.setToolTip(
            "Apply this resolution choice to every multiscale layer "
            "in the layer list. Layers with fewer levels use their "
            "coarsest available level."
        )
        layout = controls.layout()
        row, _ = layout.getWidgetPosition(level_control.level_combobox)
        layout.insertRow(row + 1, checkbox)
        checkbox.toggled.connect(self._set_apply_to_all_layers)
        self._checkboxes[layer] = checkbox

    def _set_apply_to_all_layers(self, checked):
        self.apply_to_all_layers = checked
        for checkbox in self._checkboxes.values():
            if checkbox.isChecked() != checked:
                checkbox.blockSignals(True)
                checkbox.setChecked(checked)
                checkbox.blockSignals(False)

    def _unwatch(self, layer):
        callback = self._listeners.pop(layer, None)
        if callback is not None:
            layer.events.locked_data_level.disconnect(callback)

    def _rewrite_layer(self, layer):
        animation = self.animation
        source_id = animation.layer_registry.source_id(layer)
        changed = False
        for frame in animation.key_frames:
            state = frame.viewer_state
            target = next(
                (
                    properties
                    for sid, properties in state.layers.items()
                    if sid == source_id
                    or (
                        not properties.get("source_id")
                        and properties.get("name", sid) == layer.name
                    )
                ),
                None,
            )
            if target is None:
                continue
            layers = {
                sid: (
                    dict(properties, locked_data_level=layer.locked_data_level)
                    if properties is target
                    else properties
                )
                for sid, properties in state.layers.items()
            }
            frame.viewer_state = replace(state, layers=layers)
            frame._thumbnail_pending = True
            changed = True
        if changed:
            animation._frames._rebuild_frame_index()

    def set_section_level(self, object_id, level):
        animation = self.animation
        section = next(obj for obj in animation.scene if obj.id == object_id)
        if section.kind != "ortho_slice":
            raise ValueError(
                "Resolution is only available for optical sections"
            )
        source = animation.layer_registry.find(
            section.source_id, section.source
        )
        count = (
            len(source.data) if source is not None and source.multiscale else 1
        )
        level = int(level)
        if not -count <= level < count:
            raise ValueError(f"Resolution level {level} is out of range")
        if section.channels is not None:
            for source_id, settings in section.channels.items():
                channel_source = animation.layer_registry.find(source_id)
                if channel_source is None:
                    continue
                channel_count = (
                    len(channel_source.data)
                    if channel_source.multiscale
                    else 1
                )
                if not -channel_count <= level < channel_count:
                    raise ValueError(
                        f"Resolution level {level} is out of range for "
                        f"{channel_source.name!r}"
                    )
            for settings in section.channels.values():
                settings["source_level"] = level
        section.source_level = level
        for frame in animation.key_frames:
            state = frame.viewer_state
            if not state.scene or object_id not in state.scene:
                continue
            scene = dict(state.scene)
            updated = dict(scene[object_id], source_level=level)
            if updated.get("channels") is not None:
                updated["channels"] = {
                    source_id: dict(settings, source_level=level)
                    for source_id, settings in updated["channels"].items()
                }
            scene[object_id] = updated
            frame.viewer_state = replace(state, scene=scene)
            frame._thumbnail_pending = True
        animation._frames._rebuild_frame_index()
        animation.apply_working_state()

    def close(self):
        layers = self.animation.viewer.layers
        layers.events.inserted.disconnect(self._inserted)
        layers.events.removed.disconnect(self._removed)
        for checkbox in self._checkboxes.values():
            checkbox.parentWidget().layout().removeRow(checkbox)
        self._checkboxes.clear()
        for layer in list(self._listeners):
            self._unwatch(layer)
