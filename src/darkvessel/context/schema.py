"""The contextual variable schema: columns filled in later by `darkvessel context`.

Declared once so a layer's presence and dtype never depends on whether that stage ran — see
`context.gee_layers.without_context`.
"""

CONTEXT_COLUMNS: dict[str, str] = {
    "distance_to_shore_m": "float64",
    "depth_m": "float64",
    "fishing_hours": "float64",
    "eez": "string",
}
