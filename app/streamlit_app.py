from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.inference import artifacts_available, load_artifacts  # noqa: E402
from src.io import load_events_from_bytes  # noqa: E402
from src.resolution import resolve_entities  # noqa: E402


@st.cache_resource
def get_model():
    return load_artifacts()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def main() -> None:
    st.set_page_config(page_title="Entity Resolution", page_icon="🔍", layout="wide")
    st.title("Поиск профилей одного человека")
    st.markdown(
        "Загрузите **CSV** или **Parquet** с событиями профилей. "
        "Сервис найдёт группы `profile_id`, которые, вероятно, принадлежат одному человеку."
    )

    config = load_config()
    if not artifacts_available(config["paths"]["artifacts_dir"]):
        st.error(
            "Модель не обучена. Сначала выполните:\n\n"
            "```bash\ncd ml_pipeline\ndocker compose run --rm train\n```"
        )
        st.stop()

    try:
        artifacts = get_model()
    except Exception as exc:
        st.error(f"Не удалось загрузить модель: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Параметры объединения")
        merge_mode = st.radio(
            "Режим",
            [
                "auto_merge + manual_review",
                "только auto_merge",
                "по порогу score",
            ],
            index=0,
        )
        min_score = None
        merge_actions = {"auto_merge", "manual_review"}
        if merge_mode == "только auto_merge":
            merge_actions = {"auto_merge"}
        elif merge_mode == "по порогу score":
            merge_actions = None
            min_score = st.slider("Минимальный score пары", 0.5, 1.0, 0.99, 0.01)

    uploaded = st.file_uploader(
        "Файл с событиями (.csv / .parquet)",
        type=["csv", "parquet"],
    )

    if uploaded is None:
        st.info("Загрузите файл с событиями профилей.")
        st.stop()

    if st.button("Найти профили одного человека", type="primary"):
        with st.spinner("Blocking → скоринг пар → построение кластеров…"):
            try:
                events = load_events_from_bytes(uploaded.getvalue(), uploaded.name)
                result = resolve_entities(
                    events,
                    artifacts,
                    merge_actions=merge_actions,
                    min_score=min_score,
                )
                st.session_state["result"] = result
            except Exception as exc:
                st.error(str(exc))
                st.stop()

    if "result" not in st.session_state:
        st.stop()

    result = st.session_state["result"]
    s = result.summary

    st.subheader("Результат")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Событий", s["events"])
    c2.metric("Профилей", s["profiles"])
    c3.metric("Групп (2+ профиля)", s["clusters_found"])
    c4.metric("Профилей в группах", s["profiles_in_clusters"])

    if result.clusters.empty:
        st.warning("Группы не найдены. Попробуйте другой режим или понизьте порог score.")
    else:
        st.dataframe(result.clusters, use_container_width=True, hide_index=True)
        st.download_button(
            "Скачать кластеры (CSV)",
            data=to_csv_bytes(result.clusters),
            file_name="entity_clusters.csv",
            mime="text/csv",
        )
        st.download_button(
            "Скачать пары со score (CSV)",
            data=to_csv_bytes(result.pairs_scored.sort_values("score", ascending=False)),
            file_name="scored_pairs.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
