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
from src.resolution import ResolutionResult, resolve_entities  # noqa: E402


@st.cache_resource
def get_model():
    return load_artifacts()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


PROFILE_DETAIL_COLUMNS = [
    "profile_id",
    "entity_id",
    "first_name",
    "last_name",
    "email",
    "phone",
    "birthday",
    "sex",
    "email_domain",
    "phone_digits",
    "events",
    "created_at_min",
    "created_at_max",
]


def _cluster_label(cluster_id: str, cluster_size: int) -> str:
    return f"{cluster_id} ({cluster_size} профилей)"


def render_cluster_explorer(result: ResolutionResult) -> None:
    clusters = result.clusters
    cluster_ids = sorted(clusters["cluster_id"].unique())
    sizes = clusters.groupby("cluster_id")["cluster_size"].first().to_dict()
    labels = {cid: _cluster_label(cid, sizes[cid]) for cid in cluster_ids}

    st.subheader("Просмотр кластера")
    selected_cluster = st.selectbox(
        "Выберите кластер",
        cluster_ids,
        format_func=lambda cid: labels[cid],
    )

    profile_ids = clusters.loc[
        clusters["cluster_id"] == selected_cluster, "profile_id"
    ].astype(str)
    cluster_profiles = result.profiles[
        result.profiles["profile_id"].astype(str).isin(profile_ids)
    ].copy()

    display_cols = [c for c in PROFILE_DETAIL_COLUMNS if c in cluster_profiles.columns]
    st.caption(f"Профилей в кластере: {len(cluster_profiles)}")
    st.dataframe(
        cluster_profiles[display_cols].sort_values("profile_id"),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Скачать профили кластера (CSV)",
        data=to_csv_bytes(cluster_profiles[display_cols]),
        file_name=f"{selected_cluster}_profiles.csv",
        mime="text/csv",
        key=f"download_{selected_cluster}",
    )

    profile_id_set = set(profile_ids)
    intra_pairs = result.pairs_scored[
        result.pairs_scored["left_profile_id"].astype(str).isin(profile_id_set)
        & result.pairs_scored["right_profile_id"].astype(str).isin(profile_id_set)
    ].sort_values("score", ascending=False)
    if not intra_pairs.empty:
        st.markdown("**Пары внутри кластера**")
        st.dataframe(intra_pairs, use_container_width=True, hide_index=True)


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
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Скачать кластеры (CSV)",
                data=to_csv_bytes(result.clusters),
                file_name="entity_clusters.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.download_button(
                "Скачать пары со score (CSV)",
                data=to_csv_bytes(result.pairs_scored.sort_values("score", ascending=False)),
                file_name="scored_pairs.csv",
                mime="text/csv",
            )

        st.divider()
        render_cluster_explorer(result)


if __name__ == "__main__":
    main()
