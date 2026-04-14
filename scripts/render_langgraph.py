from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.rag.graph import build_teknofest_graph


def main() -> None:
    """
    TEKNOFEST LangGraph yapısını PNG olarak çizer.

    Ortaya çıkan dosya:
      - app dizininde: app/langgraph_teknofest.png
    """
    settings = get_settings()
    graph = build_teknofest_graph(settings=settings)

    # LangGraph, dahili olarak networkx/pygraphviz üzerinden PNG çizimi sağlar.
    # Çıktıyı doğrudan app klasörü altına yaz.
    output_path = Path(__file__).resolve().parent.parent / "app" / "langgraph_teknofest.png"
    graph.get_graph().draw_png(str(output_path))
    print(f"LangGraph diagram written to: {output_path}")


if __name__ == "__main__":
    main()

