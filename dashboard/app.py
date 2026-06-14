"""EdgeMind Streamlit Dashboard — 4-page interactive benchmark visualization."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EdgeMind Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("⚡ EdgeMind")
st.sidebar.caption("LLM Optimization Toolkit")

page = st.sidebar.radio(
    "Navigation",
    ["🚀 Run Benchmarks", "📊 Tradeoff Dashboard", "🖥️ Hardware Advisor", "🆚 Groq Comparison"],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**EdgeMind v1.0.0**\n\n"
    "CLI: `edgemind --help`\n\n"
    "[GitHub](https://github.com) | [Docs](https://github.com)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIRS = [Path("./results"), Path("./sample_results")]
_COLOR_SCALE = [
    [0.0, "#d32f2f"],   # <80% red
    [0.25, "#f57c00"],  # 80-85% orange
    [0.5, "#fbc02d"],   # 85-90% yellow
    [0.75, "#388e3c"],  # 90-95% green
    [1.0, "#1b5e20"],   # >95% dark green
]


def _load_all_results() -> list[dict]:
    """Scan results dirs and load all benchmark JSON files."""
    results: list[dict] = []
    for base in RESULTS_DIRS:
        if not base.exists():
            continue
        for json_file in base.rglob("*benchmark*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                data["_source_file"] = str(json_file)
                results.append(data)
            except Exception:
                pass
    return results


def _flatten_result(r: dict) -> dict | None:
    """Flatten a nested BenchmarkResult dict into a single-level row for display."""
    try:
        row: dict = {
            "model": r.get("model_id", "unknown").split("/")[-1],
            "method": r.get("quantization_method", "unknown"),
            "size_gb": r.get("model_size_gb", 0.0),
            "compression": r.get("compression_ratio", 1.0),
            "quality_retention": r.get("quality_retention_pct"),
            "device": r.get("device", "unknown"),
            "benchmarked_at": r.get("benchmarked_at", ""),
        }
        if perplexity := r.get("perplexity"):
            row["perplexity"] = perplexity.get("mean_perplexity")
            row["perplexity_std"] = perplexity.get("std_perplexity")
        else:
            row["perplexity"] = None
        if speed := r.get("speed"):
            row["tps"] = speed.get("mean_tps")
            row["ttft_ms"] = speed.get("mean_ttft_ms")
        else:
            row["tps"] = None
            row["ttft_ms"] = None
        if memory := r.get("memory"):
            row["vram_gb"] = memory.get("peak_vram_gb")
            row["available_vram"] = memory.get("available_vram_gb")
            row["fits"] = memory.get("fits_on_device", True)
        else:
            row["vram_gb"] = None
            row["available_vram"] = None
            row["fits"] = True
        if quality := r.get("quality"):
            row["quality_score"] = quality.get("mean_quality_score")
            row["groq_score"] = quality.get("groq_comparison_score")
        else:
            row["quality_score"] = None
            row["groq_score"] = None
        return row
    except Exception:
        return None


def _retention_color(pct: float | None) -> str:
    if pct is None:
        return "gray"
    if pct >= 95:
        return "#1b5e20"
    if pct >= 90:
        return "#388e3c"
    if pct >= 85:
        return "#fbc02d"
    if pct >= 80:
        return "#f57c00"
    return "#d32f2f"


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Run Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

if page == "🚀 Run Benchmarks":
    st.title("🚀 Run Benchmarks")
    st.caption(
        "Load a model, select quantization methods, and run the full benchmark suite. "
        "Results are saved to ./results/ and appear in the Tradeoff Dashboard automatically."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        model_path = st.text_input(
            "Model path or HuggingFace ID",
            placeholder="e.g. Qwen/Qwen2.5-7B-Instruct or ./quantized/my-model",
            help="HuggingFace model ID or local directory path to the model.",
        )

    with col2:
        method = st.selectbox(
            "Quantization method",
            ["bf16", "fp16", "int8", "int4_nf4", "int4_fp4", "gptq_4bit", "awq_4bit",
             "gguf_q4_k_m", "gguf_q5_k_m", "gguf_q8_0"],
            help="The quantization format of the model being benchmarked.",
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        run_perplexity = st.checkbox("Perplexity (WikiText-2)", value=True)
        run_speed = st.checkbox("Inference Speed (TPS + TTFT)", value=True)
    with col4:
        run_memory = st.checkbox("Memory Profiler (VRAM/RAM)", value=True)
        run_quality = st.checkbox("Quality Evaluation (LLM-Judge)", value=True)
    with col5:
        compare_groq = st.toggle("Compare vs Groq API", value=False)
        device = st.selectbox("Device", ["auto", "cuda", "cpu", "mps"])

    if st.button("▶ Run Benchmarks", type="primary", disabled=not model_path):
        tests = []
        if run_memory:
            tests.append("memory")
        if run_perplexity:
            tests.append("perplexity")
        if run_speed:
            tests.append("speed")
        if run_quality:
            tests.append("quality")

        status_area = st.empty()
        progress = st.progress(0)

        try:
            from edgemind.benchmarks.runner import BenchmarkRunner
            from edgemind.models.benchmark_models import QuantizationMethod

            qm = QuantizationMethod(method)
            runner = BenchmarkRunner()

            status_area.info(f"Loading model: {model_path}...")
            progress.progress(10)

            result = runner.run_all(
                model_path=model_path,
                quantization_method=qm,
                tests=tests,
                compare_groq=compare_groq,
                device=device,
            )
            progress.progress(100)
            status_area.success("Benchmarks complete! Results saved to ./results/")

            col_a, col_b, col_c, col_d = st.columns(4)
            if result.perplexity:
                col_a.metric("Perplexity", f"{result.perplexity.mean_perplexity:.2f}")
            if result.speed:
                col_b.metric("Speed", f"{result.speed.mean_tps:.1f} tok/s")
                col_c.metric("TTFT", f"{result.speed.mean_ttft_ms:.0f}ms")
            if result.memory:
                col_d.metric("VRAM", f"{result.memory.peak_vram_gb:.1f} GB")

        except ImportError as e:
            status_area.error(f"Import error — is the package installed? {e}")
        except Exception as e:
            status_area.error(f"Benchmark failed: {e}")

    st.markdown("---")
    st.subheader("Available Sample Results")
    results = _load_all_results()
    if results:
        st.caption(f"Found {len(results)} benchmark result(s) across all directories.")
        for r in results[:5]:
            st.text(
                f"• {r.get('model_id','?').split('/')[-1]} | "
                f"{r.get('quantization_method','?')} | "
                f"{r.get('benchmarked_at','?')[:10]}"
            )
    else:
        st.info(
            "No results yet. Run benchmarks above or place JSON files in ./results/ or ./sample_results/"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Tradeoff Dashboard (THE SHOWPIECE)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Tradeoff Dashboard":
    st.title("📊 Quality vs Size vs Speed Tradeoff Dashboard")
    st.caption(
        "Each bubble = one benchmark run. Bigger bubble = better perplexity (lower is better). "
        "Color = quality retention vs Groq baseline."
    )

    raw_results = _load_all_results()
    rows = [r for r in (_flatten_result(d) for d in raw_results) if r is not None]

    if not rows:
        st.warning(
            "No benchmark results found. Run benchmarks on Page 1 or check that "
            "./sample_results/ contains JSON files."
        )
        st.stop()

    import pandas as pd

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["size_gb"])

    # ── Filters ──
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        models = sorted(df["model"].unique().tolist())
        selected_models = st.multiselect("Filter by model", models, default=models)
    with col_f2:
        methods = sorted(df["method"].unique().tolist())
        selected_methods = st.multiselect("Filter by method", methods, default=methods)

    mask = df["model"].isin(selected_models) & df["method"].isin(selected_methods)
    df_f = df[mask].copy()

    if df_f.empty:
        st.warning("No data after filtering.")
        st.stop()

    # ── Main scatter: size vs TPS, bubble=quality, color=retention ──
    st.subheader("Size vs Speed vs Quality (Bubble Chart)")

    df_scatter = df_f.dropna(subset=["tps", "size_gb"])

    if not df_scatter.empty:
        # Bubble size: inversely proportional to perplexity (bigger = better)
        df_scatter = df_scatter.copy()
        df_scatter["bubble_size"] = df_scatter["perplexity"].apply(
            lambda p: max(5, 50 - (p - 8) * 3) if p is not None else 20
        )
        df_scatter["quality_ret_display"] = df_scatter["quality_retention"].fillna(100.0)
        df_scatter["hover_text"] = df_scatter.apply(
            lambda row: (
                f"<b>{row['model']}</b><br>"
                f"Method: {row['method']}<br>"
                f"Size: {row['size_gb']:.2f} GB<br>"
                f"TPS: {row['tps']:.1f} tok/s<br>"
                + (f"Perplexity: {row['perplexity']:.2f}<br>" if row["perplexity"] else "")
                + f"Quality: {row['quality_ret_display']:.0f}%"
            ),
            axis=1,
        )

        fig = px.scatter(
            df_scatter,
            x="size_gb",
            y="tps",
            size="bubble_size",
            color="quality_ret_display",
            hover_name="model",
            hover_data={
                "method": True,
                "size_gb": ":.2f",
                "tps": ":.1f",
                "perplexity": ":.2f",
                "quality_ret_display": ":.0f",
                "bubble_size": False,
            },
            labels={
                "size_gb": "Model Size (GB)",
                "tps": "Tokens / Second",
                "quality_ret_display": "Quality Retention %",
            },
            color_continuous_scale=[
                [0, "#d32f2f"], [0.2, "#f57c00"],
                [0.5, "#fbc02d"], [0.75, "#388e3c"], [1.0, "#1b5e20"],
            ],
            range_color=[75, 100],
            text="method",
            title="LLM Quantization Tradeoff: Size ↔ Speed ↔ Quality",
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(
            height=500,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            coloraxis_colorbar=dict(
                title="Quality %",
                tickvals=[80, 85, 90, 95, 100],
            ),
            xaxis=dict(gridcolor="#333", title_font_size=13),
            yaxis=dict(gridcolor="#333", title_font_size=13),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Speed data not available — run inference speed benchmarks to populate this chart.")

    # ── Supporting charts ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Perplexity by Method (lower = better)")
        df_ppl = df_f.dropna(subset=["perplexity"]).sort_values("perplexity")
        if not df_ppl.empty:
            fig_ppl = px.bar(
                df_ppl,
                x="method",
                y="perplexity",
                color="model",
                barmode="group",
                labels={"perplexity": "Perplexity", "method": "Quantization Method"},
                title="Lower is better",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_ppl.update_layout(
                height=320,
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
                xaxis_tickangle=-30,
                showlegend=True,
            )
            fig_ppl.add_annotation(
                text="↑ Worse", x=0.01, y=0.99, xref="paper", yref="paper",
                showarrow=False, font=dict(color="red", size=10)
            )
            fig_ppl.add_annotation(
                text="↓ Better", x=0.01, y=0.01, xref="paper", yref="paper",
                showarrow=False, font=dict(color="green", size=10)
            )
            st.plotly_chart(fig_ppl, use_container_width=True)
        else:
            st.info("No perplexity data available.")

    with col_right:
        st.subheader("Quality Retention vs Compression Ratio")
        df_ret = df_f.dropna(subset=["quality_retention", "compression"])
        if not df_ret.empty:
            fig_ret = px.line(
                df_ret.sort_values("compression"),
                x="compression",
                y="quality_retention",
                color="model",
                markers=True,
                labels={
                    "compression": "Compression Ratio (x)",
                    "quality_retention": "Quality Retention (%)",
                },
                title="Quality degradation as compression increases",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig_ret.add_hline(
                y=90, line_dash="dash", line_color="orange",
                annotation_text="90% threshold (recommended minimum)"
            )
            fig_ret.update_layout(
                height=320,
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
            )
            st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.info("No quality retention data available.")

    # ── Summary sortable table ──
    st.subheader("Benchmark Summary Table")

    display_cols = {
        "model": "Model",
        "method": "Method",
        "size_gb": "Size (GB)",
        "perplexity": "Perplexity ↓",
        "tps": "TPS ↑",
        "quality_score": "Quality /10",
        "quality_retention": "vs Groq %",
        "vram_gb": "VRAM (GB)",
        "fits": "Fits RTX5090",
    }

    table_df = df_f[list(display_cols.keys())].rename(columns=display_cols).copy()

    def _color_row(row: object) -> list[str]:
        try:
            q = row["vs Groq %"]  # type: ignore[index]
            q = None if q != q else q  # NaN check
        except (KeyError, TypeError):
            q = None
        if q is None:
            color = ""
        elif q >= 90:
            color = "background-color: #1b5e2033"
        elif q >= 80:
            color = "background-color: #fbc02d22"
        else:
            color = "background-color: #d32f2f22"
        return [color] * len(row)

    styled = table_df.style.apply(_color_row, axis=1).format(
        {
            "Size (GB)": "{:.2f}",
            "Perplexity ↓": lambda x: f"{x:.2f}" if x == x else "—",
            "TPS ↑": lambda x: f"{x:.1f}" if x == x else "—",
            "Quality /10": lambda x: f"{x:.1f}" if x == x else "—",
            "vs Groq %": lambda x: f"{x:.0f}%" if x == x else "—",
            "VRAM (GB)": lambda x: f"{x:.1f}" if x == x else "—",
        }
    )
    st.dataframe(styled, use_container_width=True, height=400)

    # ── Key metrics summary ──
    st.markdown("---")
    st.subheader("Key Insight")
    best_method = df_f.sort_values("quality_retention", ascending=False).iloc[0] if not df_f.empty else None
    if best_method is not None:
        retention = best_method.get("quality_retention") or 100.0
        size = best_method.get("size_gb", 0)
        method_name = best_method.get("method", "?")
        st.success(
            f"**Best quality/size tradeoff: {method_name}** — "
            f"{retention:.0f}% quality retention at {size:.1f} GB. "
            f"Recommended for production edge deployment."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Hardware Advisor
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🖥️ Hardware Advisor":
    st.title("🖥️ Hardware Advisor")
    st.caption(
        "Select your target hardware and model size to get an instant quantization recommendation, "
        "VRAM estimate, and setup instructions."
    )

    try:
        from edgemind.deployment.profiles import PROFILE_DISPLAY_NAMES, get_profile
        from edgemind.core.gpu_utils import GPUInfo
        from edgemind.models.benchmark_models import QuantizationMethod

        col1, col2 = st.columns(2)

        with col1:
            hw_key = st.selectbox(
                "Target hardware",
                list(PROFILE_DISPLAY_NAMES.keys()),
                format_func=lambda k: PROFILE_DISPLAY_NAMES[k],
                index=0,
            )

        with col2:
            params = st.slider(
                "Model size (billions of parameters)",
                min_value=1.0,
                max_value=70.0,
                value=7.0,
                step=0.5,
                help="Number of parameters in the model you want to run.",
            )

        hw_profile = get_profile(hw_key)
        rec = hw_profile.get_recommendation(params)
        method_str = rec.get("method", "int4_nf4")

        try:
            qm = QuantizationMethod(method_str)
            gpu_utils = GPUInfo()
            required_gb = gpu_utils.estimate_vram_requirement(params, qm)
        except ValueError:
            required_gb = params * 0.5

        available = hw_profile.vram_gb if hw_profile.vram_gb > 0 else hw_profile.ram_gb
        fits = required_gb <= available

        st.markdown("---")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Recommended Method", method_str.upper())
        with col_b:
            st.metric("Required Memory", f"{required_gb:.1f} GB")
        with col_c:
            st.metric("Available Memory", f"{available:.1f} GB")

        if fits:
            st.success(f"✓ **{params:.0f}B model fits on {hw_profile.name}** — {method_str.upper()} recommended.")
        else:
            st.error(
                f"✗ **{params:.0f}B model at {method_str.upper()} exceeds {available:.0f}GB.** "
                f"Try a smaller model or more aggressive quantization."
            )

        col_d, col_e = st.columns(2)
        with col_d:
            st.info(f"**Expected throughput:** {rec.get('expected_tps', 'N/A')} tokens/second")
        with col_e:
            st.info(f"**Inference backend:** {hw_profile.recommended_inference}")

        st.markdown(f"**Notes:** {rec.get('notes', '')}")

        # VRAM comparison across all methods
        st.subheader("VRAM Requirements by Quantization Method")
        methods_check = [
            ("BF16", QuantizationMethod.BF16),
            ("FP16", QuantizationMethod.FP16),
            ("INT8", QuantizationMethod.INT8),
            ("INT4 NF4", QuantizationMethod.INT4_NF4),
            ("GPTQ 4-bit", QuantizationMethod.GPTQ_4BIT),
            ("AWQ 4-bit", QuantizationMethod.AWQ_4BIT),
            ("GGUF Q5_K_M", QuantizationMethod.GGUF_Q5KM),
            ("GGUF Q4_K_M", QuantizationMethod.GGUF_Q4KM),
            ("GGUF Q3_K_M", QuantizationMethod.GGUF_Q3KM),
            ("GGUF Q2_K", QuantizationMethod.GGUF_Q2K),
        ]
        vram_data = []
        for label, qm_val in methods_check:
            req = GPUInfo().estimate_vram_requirement(params, qm_val)
            vram_data.append({
                "Method": label,
                "Required GB": req,
                "Fits": "✓" if req <= available else "✗",
                "Color": "#388e3c" if req <= available else "#d32f2f",
            })

        import pandas as pd

        vram_df = pd.DataFrame(vram_data)
        fig_vram = go.Figure(
            go.Bar(
                x=vram_df["Required GB"],
                y=vram_df["Method"],
                orientation="h",
                marker_color=vram_df["Color"],
                text=[f"{v:.1f} GB {f}" for v, f in zip(vram_df["Required GB"], vram_df["Fits"])],
                textposition="outside",
            )
        )
        fig_vram.add_vline(
            x=available, line_dash="dash", line_color="yellow",
            annotation_text=f"{hw_profile.name} limit ({available:.0f}GB)",
        )
        fig_vram.update_layout(
            height=380,
            xaxis_title="VRAM Required (GB)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            title=f"Memory requirements for {params:.0f}B model on {hw_profile.name}",
        )
        st.plotly_chart(fig_vram, use_container_width=True)

        # Setup instructions
        with st.expander("Setup Instructions"):
            st.code(hw_profile.setup_notes, language="bash")

        # Recommended HuggingFace models
        st.subheader("Recommended Models")
        model_recs = _get_model_recommendations(params, hw_profile.name)
        for m in model_recs:
            st.markdown(f"- **{m['name']}** ({m['params']}) — {m['notes']}")

    except ImportError as e:
        st.error(f"EdgeMind not installed: {e}. Run `pip install -e .` from the project root.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Groq Comparison
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🆚 Groq Comparison":
    st.title("🆚 Local Model vs Groq API Comparison")
    st.caption(
        "Compare your local quantized model side-by-side with Groq's full-precision "
        "llama-3.3-70b baseline. Quantifies exactly how much quality you trade for "
        "speed, privacy, and zero cost."
    )

    raw_results = _load_all_results()
    result_labels = [
        f"{r.get('model_id','?').split('/')[-1]} / {r.get('quantization_method','?')} "
        f"({r.get('benchmarked_at','?')[:10]})"
        for r in raw_results
    ]

    if not result_labels:
        st.info(
            "No benchmark results found. Run benchmarks on Page 1 first, "
            "or place JSON files in ./sample_results/."
        )
        st.stop()

    selected_idx = st.selectbox(
        "Select benchmark result to analyze",
        range(len(result_labels)),
        format_func=lambda i: result_labels[i],
    )

    selected = raw_results[selected_idx]
    quality = selected.get("quality", {})

    # ── Key metrics ──
    col1, col2, col3, col4 = st.columns(4)
    local_score = quality.get("mean_quality_score") or 0
    groq_score = quality.get("groq_comparison_score") or 0
    retention = quality.get("quality_retention_pct") or (
        round(local_score / groq_score * 100, 1) if groq_score > 0 else None
    )

    col1.metric("Local Quality Score", f"{local_score:.1f}/10" if local_score else "—")
    col2.metric("Groq Baseline Score", f"{groq_score:.1f}/10" if groq_score else "—")
    col3.metric("Quality Retention", f"{retention:.0f}%" if retention else "—")
    speed = selected.get("speed", {})
    col4.metric("Local Speed", f"{speed.get('mean_tps', 0):.1f} tok/s" if speed else "—")

    # ── Verdict box ──
    if retention is not None:
        if retention >= 95:
            verdict_color = "success"
            verdict = f"Excellent — {retention:.0f}% quality at $0.00/query. No meaningful degradation."
        elif retention >= 90:
            verdict_color = "success"
            verdict = f"Good — {retention:.0f}% quality retained. Suitable for most workloads."
        elif retention >= 85:
            verdict_color = "warning"
            verdict = f"Acceptable — {retention:.0f}% quality. Consider INT8 or GGUF Q5_K_M for better results."
        else:
            verdict_color = "error"
            verdict = f"Poor — only {retention:.0f}% quality. Use less aggressive quantization."

        getattr(st, verdict_color)(f"**Verdict:** {verdict}")

    # ── Per-category breakdown ──
    categories = quality.get("prompt_categories", {})
    if categories:
        st.subheader("Quality Score by Category")
        import pandas as pd

        cat_df = pd.DataFrame(
            [{"Category": k.title(), "Score": v} for k, v in categories.items()]
        )
        fig_cat = px.bar(
            cat_df,
            x="Category",
            y="Score",
            color="Score",
            color_continuous_scale=["#d32f2f", "#fbc02d", "#1b5e20"],
            range_color=[5, 10],
            range_y=[0, 10],
            labels={"Score": "Quality Score (0-10)"},
            title="LLM-as-Judge scores by prompt category",
        )
        fig_cat.add_hline(
            y=8.0, line_dash="dash", line_color="white",
            annotation_text="Good threshold (8/10)"
        )
        fig_cat.update_layout(
            height=320,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Cost analysis ──
    st.subheader("Cost & Speed Analysis")
    col_cost1, col_cost2 = st.columns(2)

    with col_cost1:
        st.markdown("**Local Inference (EdgeMind)**")
        st.markdown(
            "- Cost: **$0.00** (hardware amortized)\n"
            "- Privacy: **100%** (data never leaves your machine)\n"
            "- Latency: No network round-trip\n"
            "- Offline: **Always available**\n"
            f"- Speed: **{speed.get('mean_tps', 0):.0f} tok/s** on your hardware"
        )

    with col_cost2:
        st.markdown("**Groq API (llama-3.3-70b)**")
        st.markdown(
            "- Cost: ~$0.0006-0.0012 per query\n"
            "- Privacy: Queries sent to Groq servers\n"
            "- Latency: ~100-300ms network overhead\n"
            "- Offline: Requires internet connection\n"
            "- Speed: ~500-700 tok/s (API throughput)"
        )

    # ── Full result JSON (expandable) ──
    with st.expander("Full Result JSON"):
        st.json(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: model recommendations
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_recommendations(params: float, hw_name: str) -> list[dict]:
    """Return a list of recommended HuggingFace models for a given size and hardware."""
    if params <= 2:
        return [
            {"name": "TinyLlama-1.1B-Chat", "params": "1.1B", "notes": "Fast, minimal footprint. Great for IoT."},
            {"name": "Qwen2.5-1.5B-Instruct", "params": "1.5B", "notes": "Excellent quality for size."},
            {"name": "SmolLM2-1.7B-Instruct", "params": "1.7B", "notes": "State-of-the-art tiny model."},
        ]
    if params <= 4:
        return [
            {"name": "Phi-4-Mini-Instruct", "params": "3.8B", "notes": "Microsoft's best small model. SOTA reasoning."},
            {"name": "Llama-3.2-3B-Instruct", "params": "3B", "notes": "Meta's compact but capable model."},
            {"name": "Qwen2.5-3B-Instruct", "params": "3B", "notes": "Strong coding and instruction following."},
        ]
    if params <= 8:
        return [
            {"name": "Qwen2.5-7B-Instruct", "params": "7B", "notes": "Top 7B model. Excellent coding."},
            {"name": "Llama-3.1-8B-Instruct", "params": "8B", "notes": "Meta's flagship 8B. Well-rounded."},
            {"name": "Mistral-7B-Instruct-v0.3", "params": "7B", "notes": "Efficient European model."},
        ]
    if params <= 14:
        return [
            {"name": "Qwen2.5-14B-Instruct", "params": "14B", "notes": "Best 14B overall. Top benchmark scores."},
            {"name": "Llama-3.1-8B-Instruct", "params": "8B", "notes": "Slightly smaller but well-optimized."},
        ]
    return [
        {"name": "Qwen2.5-32B-Instruct", "params": "32B", "notes": "Best open 32B model."},
        {"name": "Llama-3.3-70B-Instruct", "params": "70B", "notes": "Top open 70B, needs INT4."},
        {"name": "DeepSeek-V3", "params": "67B", "notes": "Exceptional quality for coding tasks."},
    ]
