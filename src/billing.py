from __future__ import annotations

from typing import Any

from .config import PRICE_MAP


def summarise_bill(per_plate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {k: 0 for k in PRICE_MAP}
    for row in per_plate_rows:
        colour = row.get("colour_class", "unknown")
        if colour not in PRICE_MAP:
            colour = "unknown"
        counts[colour] += 1

    breakdown: dict[str, dict[str, float]] = {}
    total = 0.0
    for colour, unit in PRICE_MAP.items():
        n = counts.get(colour, 0)
        sub = round(n * unit, 2)
        breakdown[colour] = {
            "count": n,
            "unit_price": unit,
            "subtotal": sub,
        }
        total += sub

    return {
        **breakdown,
        "total": round(total, 2),
        "plate_count": len(per_plate_rows),
    }


def bill_summary_markdown(bill: dict[str, Any]) -> str:
    lines = [
        "| type | count | unit price | subtotal |",
        "| ---- | ----: | ---------: | -------: |",
    ]
    for colour in ("white", "black", "gold", "unknown"):
        row = bill[colour]
        lines.append(
            f"| {colour} | {row['count']} | ${row['unit_price']:.2f} | ${row['subtotal']:.2f} |"
        )
    lines.append(f"| **total** | | | **${bill['total']:.2f}** |")
    return "\n".join(lines)
