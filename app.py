import streamlit as st
import subprocess
import json
import os
import time
from datetime import datetime
import pandas as pd
from pathlib import Path
from urllib.request import urlopen
from functools import lru_cache

# Compact HTML template location (repo-local)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
COMPACT_TEMPLATE_PATH = TEMPLATE_DIR / "benchmark_compact_template.html"

# Local .env loader (keeps dependencies minimal)
def _load_dotenv(path=".env"):
    env_data = {}
    env_path = Path(path)
    if not env_path.exists():
        return env_data
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env_data[key] = value
    return env_data


_DOTENV = _load_dotenv()


def _env_default(key, fallback=""):
    return os.getenv(key) or _DOTENV.get(key, fallback)


def _find_latest_file(output_dir, patterns):
    output_path = Path(output_dir)
    matches = []
    for pattern in patterns:
        matches.extend(output_path.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _is_report_template_valid(content: str) -> bool:
    if not content:
        return False
    lower = content.lower()
    if "page not found" in lower or "<title>page not found" in lower or ">404<" in lower:
        return False
    # Heuristic: GuideLLM report UI should mention GuideLLM and workload/report sections
    if "guidellm" in lower and ("workload report" in lower or "benchmark" in lower):
        return True
    # Fallback: accept if it looks like a UI bundle (Next.js) but not a 404 page
    return "next" in lower and "guidellm" in lower


def _ensure_local_report_template(source_url: str) -> Path | None:
    app_root = Path(__file__).resolve().parent
    cache_dir = app_root / ".guidellm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "guidellm_report_template.html"
    if local_path.exists():
        try:
            existing = local_path.read_text(encoding="utf-8")
            if _is_report_template_valid(existing):
                return local_path
        except Exception:
            pass
    try:
        with urlopen(source_url, timeout=30) as response:
            content = response.read()
        local_path.write_bytes(content)
        try:
            if _is_report_template_valid(local_path.read_text(encoding="utf-8")):
                return local_path
        except Exception:
            pass
    except Exception:
        return None


@lru_cache
def _guidellm_cli_help() -> str:
    try:
        result = subprocess.run(
            ["guidellm", "benchmark", "--help"],
            capture_output=True,
            text=True,
            check=False
        )
        return (result.stdout or "") + "\n" + (result.stderr or "")
    except Exception:
        return ""


def _guidellm_cli_capabilities() -> dict:
    help_text = _guidellm_cli_help()
    if not help_text.strip():
        # Default to newer CLI behavior if we cannot detect capabilities.
        return {
            "profile": True,
            "rate_type": False,
            "outputs": True,
            "output_dir": True,
            "output_path": False,
            "backend_kwargs": True,
            "backend_args": False,
            "output_extras": True,
            "sample_requests": True,
            "rampup": True,
            "detect_saturation": True
        }
    return {
        "profile": "--profile" in help_text,
        "rate_type": "--rate-type" in help_text,
        "outputs": "--outputs" in help_text,
        "output_dir": "--output-dir" in help_text,
        "output_path": "--output-path" in help_text,
        "backend_kwargs": "--backend-kwargs" in help_text,
        "backend_args": "--backend-args" in help_text,
        "output_extras": "--output-extras" in help_text,
        "sample_requests": "--sample-requests" in help_text,
        "rampup": "--rampup" in help_text,
        "detect_saturation": "--detect-saturation" in help_text
    }


def _safe_get(data, path, default=None):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _format_number(val, digits=2):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return str(val)


def _format_meta_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        # redact common secrets
        redacted = {}
        for k, v in value.items():
            if "key" in k.lower() or "token" in k.lower():
                redacted[k] = "****"
            else:
                redacted[k] = v
        return json.dumps(redacted)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _extract_compact_metrics(report: dict) -> dict:
    benchmarks = report.get("benchmarks") or []
    if not benchmarks:
        return {}
    bench = benchmarks[0]
    metrics = bench.get("metrics", {})

    rps_mean = _safe_get(metrics, ["requests_per_second", "successful", "mean"])
    conc_mean = _safe_get(metrics, ["request_concurrency", "successful", "mean"])
    out_tps_mean = _safe_get(metrics, ["output_tokens_per_second", "successful", "mean"])
    total_tps_mean = _safe_get(metrics, ["tokens_per_second", "successful", "mean"])

    lat_mean_s = _safe_get(metrics, ["request_latency", "successful", "mean"])
    lat_median_s = _safe_get(metrics, ["request_latency", "successful", "median"])
    lat_p99_s = _safe_get(metrics, ["request_latency", "successful", "percentiles", "p99"])

    lat_mean_ms = None if lat_mean_s is None else float(lat_mean_s) * 1000
    lat_median_ms = None if lat_median_s is None else float(lat_median_s) * 1000
    lat_p99_ms = None if lat_p99_s is None else float(lat_p99_s) * 1000

    ttft_mean = _safe_get(metrics, ["time_to_first_token_ms", "successful", "mean"])
    ttft_median = _safe_get(metrics, ["time_to_first_token_ms", "successful", "median"])
    ttft_p99 = _safe_get(metrics, ["time_to_first_token_ms", "successful", "percentiles", "p99"])

    args = report.get("args") or {}
    model = _safe_get(report, ["args", "model"]) or _safe_get(report, ["metadata", "model"]) or "N/A"
    strategy = bench.get("type_") or _safe_get(bench, ["config", "strategy_type"]) or "N/A"

    return {
        "model": model,
        "strategy": strategy,
        "meta": [
            ("Target", _format_meta_value(args.get("target"))),
            ("Profile", _format_meta_value(args.get("profile"))),
            ("Rate", _format_meta_value(args.get("rate"))),
            ("Max Seconds", _format_meta_value(args.get("max_seconds"))),
            ("Max Requests", _format_meta_value(args.get("max_requests"))),
            ("Data", _format_meta_value(args.get("data"))),
            ("Outputs", _format_meta_value(args.get("outputs"))),
            ("Sample Requests", _format_meta_value(args.get("sample_requests"))),
            ("Rampup", _format_meta_value(args.get("rampup"))),
        ],
        "rows": [
            ("Requests/Second", _format_number(rps_mean)),
            ("Concurrency", _format_number(conc_mean)),
            ("Output Tokens/sec", _format_number(out_tps_mean)),
            ("Total Tokens/sec", _format_number(total_tps_mean)),
            ("Mean Latency (ms)", _format_number(lat_mean_ms)),
            ("Median Latency (ms)", _format_number(lat_median_ms)),
            ("P99 Latency (ms)", _format_number(lat_p99_ms)),
            ("Mean TTFT (ms)", _format_number(ttft_mean)),
            ("Median TTFT (ms)", _format_number(ttft_median)),
            ("P99 TTFT (ms)", _format_number(ttft_p99)),
        ],
    }


def _extract_quick_stats(report: dict) -> dict:
    data = _extract_compact_metrics(report)
    if not data:
        return {}
    rows = {name: value for name, value in data["rows"]}
    return {
        "requests_per_sec": rows.get("Requests/Second"),
        "tokens_per_sec": rows.get("Total Tokens/sec"),
        "latency_ms": rows.get("Mean Latency (ms)"),
        "ttft_ms": rows.get("Mean TTFT (ms)"),
    }


def _render_compact_report(
    report: dict,
    output_dir: Path,
    overwrite_default: bool = True,
    theme: str | None = None,
) -> Path | None:
    if not report or not COMPACT_TEMPLATE_PATH.exists():
        return None
    data = _extract_compact_metrics(report)
    if not data:
        return None
    template = COMPACT_TEMPLATE_PATH.read_text(encoding="utf-8")
    rows_html = "\n".join(
        f"<tr><td>{name}</td><td class=\"value\">{value}</td></tr>"
        for name, value in data["rows"]
    )
    meta_rows_html = "\n".join(
        f"<tr><td>{name}</td><td>{value}</td></tr>"
        for name, value in data.get("meta", [])
    )
    html = (
        template.replace("{{MODEL}}", str(data["model"]))
        .replace("{{STRATEGY}}", str(data["strategy"]))
        .replace("{{ROWS}}", rows_html)
        .replace("{{META_ROWS}}", meta_rows_html)
        .replace("{{THEME_CLASS}}", f"theme-{theme}" if theme else "theme-dark")
    )
    out_path = output_dir / "benchmarks_compact.html"
    out_path.write_text(html, encoding="utf-8")
    if overwrite_default:
        (output_dir / "benchmarks.html").write_text(html, encoding="utf-8")
    return out_path

# Try to import optional dependencies
try:
    import yaml
except ImportError:
    st.error("PyYAML not installed. Run: pip install pyyaml")
    st.stop()

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="GuideLLM Benchmark Workbench",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 GuideLLM Benchmark Workbench")
st.markdown("*A user-friendly interface for running GuideLLM benchmarks*")

# Initialize session state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'current_output' not in st.session_state:
    st.session_state.current_output = ""
if 'benchmark_running' not in st.session_state:
    st.session_state.benchmark_running = False

CLI_CAPS = _guidellm_cli_capabilities()

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("Defaults are loaded from `.env` for `TARGET_URL`, `API_KEY`, and `HF_TOKEN` when available.")
    st.caption("HTML report template defaults to local `.guidellm_cache/` unless `GUIDELLM__REPORT_GENERATION__SOURCE` is set.")
    if CLI_CAPS.get("rate_type") and not CLI_CAPS.get("profile"):
        st.info("Detected GuideLLM CLI without `--profile`. Using `--rate-type` internally for compatibility.")
    if not CLI_CAPS.get("outputs") and CLI_CAPS.get("output_path"):
        st.info("Detected older GuideLLM CLI without `--outputs`. JSON output will be generated via `--output-path`, and HTML will be rendered by the app.")
    
    # Basic Parameters
    st.subheader("Endpoint Configuration")
    
    backend_type = st.selectbox(
        "Backend Type",
        ["OpenAI-Compatible", "LiteLLM Proxy"],
        index=1,
        help="Select the backend type. LiteLLM uses OpenAI-compatible endpoints but often requires a specific model string."
    )

    target = st.text_input(
        "Target Endpoint", 
        value=_env_default("TARGET_URL", "http://localhost:8000"),
        placeholder="http://your-endpoint:8000",
        help="Specifies the target path for the backend server to run benchmarks against. Use the base URL for OpenAI-compatible servers (add /v1 if required by your proxy)."
    )
    
    model_name = st.text_input(
        "Model Name",
        value="llama-3-2-3b",
        placeholder="Enter model identifier",
        help="Allows selecting a specific model from the server. If not provided, defaults to the first model available. Useful when multiple models are hosted on the same endpoint."
    )

    litellm_model = None
    if backend_type == "LiteLLM Proxy":
        litellm_model = st.text_input(
            "LiteLLM Model Name",
            value="",
            placeholder="e.g., openai/gpt-4o-mini, anthropic/claude-3-5-sonnet",
            help="Model identifier as expected by LiteLLM. If provided, this overrides Model Name for the benchmark."
        )
    
    # Authentication
    st.subheader("Authentication")
    
    api_key = st.text_input(
        "API Key",
        type="password",
        value=_env_default("API_KEY", ""),
        placeholder="Your API key (optional)",
        help="Authentication key for accessing the target endpoint. Passed to GuideLLM via --backend-kwargs as api_key, which sets Authorization: Bearer <api_key>."
    )
    
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password", 
        value=_env_default("HF_TOKEN", ""),
        placeholder="For gated models (optional)",
        help="Authentication token for accessing gated Hugging Face models. Required only when using gated/private models from Hugging Face. Visit HuggingFace Settings to create a token."
    )
    
    # Advanced Parameters
    st.subheader("Benchmark Parameters")
    
    profile = st.selectbox(
        "Profile (Rate Type)",
        ["synchronous", "throughput", "concurrent", "constant", "poisson", "sweep"],
        index=0,
        help="Benchmark profile (strategy). synchronous: sequential. throughput: maximum throughput. concurrent: fixed concurrency. constant: fixed rate. poisson: rate via Poisson distribution. sweep: baseline+throughput+interpolated constants."
    )
    
    # Rate parameter (required for some rate types)
    rate = None
    if profile == "throughput":
        rate = st.number_input(
            "Rate (concurrent streams)",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
            help="Number of concurrent request streams for throughput profile."
        )
    elif profile in ["concurrent", "constant", "poisson"]:
        rate = st.text_input(
            "Rate(s)",
            value="10",
            help="Single value or comma-separated list (e.g., 10 or 10,20,30). For concurrent: streams. For constant/poisson: req/s."
        )
    elif profile == "sweep":
        rate = st.number_input(
            "Sweep Strategies Count",
            min_value=3,
            max_value=50,
            value=10,
            step=1,
            help="Number of strategies to run in sweep (includes synchronous + throughput)."
        )

    rampup = None
    if profile in ["throughput", "concurrent", "constant", "sweep"]:
        rampup = st.number_input(
            "Rampup (seconds)",
            min_value=0,
            max_value=3600,
            value=10,
            step=1,
            help="Duration to ramp up to target rate for throughput/constant/concurrent/sweep."
        )
    
    max_seconds = st.number_input(
        "Max Duration (seconds)",
        min_value=10,
        max_value=3600,
        value=60,
        step=10,
        help="Sets the maximum duration for each benchmark run. Benchmark stops when either duration OR request limit is reached. Typical values: 30-300s for quick tests, 600+ for production validation."
    )
    
    max_requests = st.number_input(
        "Max Requests",
        min_value=1,
        max_value=10000,
        value=100,
        step=10,
        help="Sets the maximum number of requests for each benchmark run. If not provided, runs until max-seconds is reached or dataset exhausted. Useful for consistent test sizes. Typical: 100-1000 for quick tests, 5000+ for thorough benchmarks."
    )
    
    max_concurrency = st.number_input(
        "Max Concurrency (Env Override)",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Sets GUIDELLM__MAX_CONCURRENCY env var (if supported by your GuideLLM version). Useful for global caps on concurrency."
    )

    disable_progress = st.checkbox(
        "Disable Progress Output",
        value=False,
        help="Disables progress rendering in the console."
    )

    disable_console_outputs = st.checkbox(
        "Disable Console Output",
        value=False,
        help="Disables all console output from GuideLLM."
    )

    detect_saturation = st.checkbox(
        "Detect Saturation",
        value=False,
        help="Enables saturation detection during benchmarking."
    )
    
    # Processor Configuration  
    st.subheader("Processor/Tokenizer")
    
    processor_type = st.selectbox(
        "Processor Type", 
        ["Custom processor", "Use model default"], 
        index=1,
        help="Determines how tokenization is handled for synthetic data creation. Custom processor: Specify a HuggingFace model ID or local path. Use model default: Uses a lightweight default processor (gpt2)."
    )
    
    if processor_type == "Custom processor":
        processor = st.text_input(
            "Processor Path",
            value="",
            placeholder="e.g., meta-llama/Llama-3.2-3B, gpt2, microsoft/DialoGPT-medium",
            help="HuggingFace model ID or local path to processor/tokenizer. Must match the model's processor/tokenizer for accuracy. Used for synthetic data creation and local token metrics. Supports both HuggingFace IDs and local paths."
        )
    else:
        processor = None  # Let GuideLLM default to model or backend-provided processor

    # Processor resolution: always prefer explicit processor, otherwise use Model Name
    processor_effective = processor or (model_name if model_name else None)
    
    # Data Configuration
    st.subheader("Data Configuration")
    
    data_source = st.selectbox(
        "Dataset Source", 
        ["Synthetic", "HuggingFace Dataset", "Local File/Directory", "Custom JSON"], 
        index=1,
        help="Specifies the dataset source for benchmark requests."
    )
    
    data_config = None

    if data_source == "Synthetic":
        st.info("💡 Synthetic data generates prompts/outputs with specified token counts.")
        if processor_effective and ("llama" in processor_effective.lower() or "meta-llama" in processor_effective.lower()):
            st.warning("🔒 **Gated Model Detected**: Llama models require a HuggingFace token. Make sure to provide your HF token above.")
        
        if processor_type == "Use model default" or not processor_effective:
            st.warning("⚠️ **Synthetic data works best with a Custom Processor.** Consider specifying a HuggingFace model ID for better token accuracy.")
        
        prompt_tokens = st.number_input(
            "Prompt Tokens",
            min_value=1,
            max_value=4096,
            value=512,
            step=1,
            help="Average number of tokens for input prompts."
        )
        
        output_tokens = st.number_input(
            "Output Tokens", 
            min_value=1,
            max_value=2048,
            value=128,
            step=1,
            help="Average number of tokens for generated outputs."
        )

        samples = st.number_input(
            "Samples",
            min_value=1,
            max_value=100000,
            value=1000,
            step=100,
            help="Number of synthetic samples to generate."
        )
        
        data_config = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "samples": samples
        }
    elif data_source == "HuggingFace Dataset":
        hf_dataset = st.text_input(
            "HuggingFace Dataset Path",
            value="",
            placeholder="e.g., garage-bAInd/Open-Platypus",
            help="Hugging Face dataset ID or local dataset directory."
        )
        data_config = hf_dataset
    elif data_source == "Local File/Directory":
        local_data_path = st.text_input(
            "Local Dataset Path",
            value="",
            placeholder="/path/to/dataset.jsonl",
            help="Local path to dataset file or directory."
        )
        data_config = local_data_path
    else:
        custom_data = st.text_area(
            "Custom Data Config (JSON)",
            value='{"prompt_tokens": 512, "output_tokens": 128}',
            height=120,
            help="JSON string passed to --data. Useful for advanced synthetic or in-memory configs."
        )
        try:
            data_config = json.loads(custom_data)
        except:
            st.error("Invalid JSON format")
            data_config = "prompt_tokens=512,output_tokens=128"

    with st.expander("Advanced Data Options", expanded=False):
        data_args = st.text_area(
            "Data Args (JSON)",
            value="",
            height=80,
            help="Optional JSON for --data-args (e.g., {\"split\": \"train\", \"prompt_column\": \"text\"})."
        )
        data_sampler = st.selectbox(
            "Data Sampler",
            ["none", "random"],
            index=0,
            help="Optional sampling strategy for datasets."
        )
        processor_args = st.text_area(
            "Processor Args (JSON)",
            value="",
            height=80,
            help="Optional JSON for --processor-args."
        )
    
    # Output Format Selection
    st.subheader("Output Format")
    
    output_formats = st.multiselect(
        "Output Formats",
        ["json", "csv", "html", "yaml"],
        default=["html", "json"],
        help="Select output formats to generate. HTML is required for in-app report rendering."
    )

    sample_requests = st.number_input(
        "Sample Requests (Optional)",
        min_value=0,
        max_value=1000,
        value=20,
        step=5,
        help="Limits the number of request samples stored in output files. Set to 0 to disable sampling."
    )

    output_extras = st.text_area(
        "Output Extras (JSON)",
        value="",
        height=80,
        help="Optional JSON for --output-extras (tags, metadata, etc.)."
    )

    backend_kwargs_raw = st.text_area(
        "Backend Kwargs (JSON)",
        value="",
        height=80,
        help="Optional JSON passed to --backend-kwargs (merged with api_key if provided)."
    )

    st.subheader("Display Options")
    show_html_report = st.checkbox(
        "Show HTML Report",
        value=False,
        help="Toggle inline HTML report rendering in the app."
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Run Benchmark")
    
    # Display current configuration
    with st.expander("📋 Current Configuration", expanded=False):
        effective_model = litellm_model if litellm_model else model_name
        config_display = {
            "Target": target,
            "Model": effective_model if effective_model else "(auto)",
            "Backend Type": backend_type,
            "Processor": processor_effective if processor_effective else "Model default",
            "Profile": profile,
            "Rate": rate if rate else "Auto",
            "Rampup": f"{rampup}s" if rampup is not None else "Auto",
            "Max Duration": f"{max_seconds}s",
            "Max Requests": max_requests,
            "Max Concurrency": max_concurrency,
            "Data Config": data_config,
            "Output Formats": output_formats
        }
        st.json(config_display)
    
    # Run benchmark button
    if st.button("🚀 Run Benchmark", type="primary", disabled=st.session_state.benchmark_running):
        # Clear any previous live metrics
        if 'live_metrics' in st.session_state:
            del st.session_state.live_metrics
        if 'final_benchmark_results' in st.session_state:
            del st.session_state.final_benchmark_results
        
        # Validation checks
        validation_errors = []
        if not target:
            validation_errors.append("Please provide a target endpoint")

        if backend_type == "LiteLLM Proxy" and not (litellm_model or model_name):
            validation_errors.append("LiteLLM requires a model name (LiteLLM Model Name or Model Name)")

        if data_source == "HuggingFace Dataset" and not data_config:
            validation_errors.append("Please provide a HuggingFace dataset path")

        if data_source == "Local File/Directory" and not data_config:
            validation_errors.append("Please provide a local dataset path")

        if data_source == "Synthetic" and not processor_effective:
            validation_errors.append(
                "Synthetic data requires a tokenizer/processor. Set Processor Path to a HuggingFace model ID (e.g., Qwen/Qwen2.5-0.5B-Instruct) or a local tokenizer path."
            )
        
        # Check for gated models without tokens
        if processor and ("llama" in processor.lower() or "meta-llama" in processor.lower()) and not hf_token:
            validation_errors.append("Llama models require a HuggingFace token. Please provide your HF token in the Authentication section.")
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            st.session_state.benchmark_running = True
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            effective_model = litellm_model if litellm_model else model_name
            model_slug = (effective_model if effective_model else "benchmark").replace("/", "-").replace(" ", "_")
            output_dir = Path(f"./results/{model_slug}_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure HTML is generated for in-app rendering when supported
            outputs_final = list(output_formats) if output_formats else []
            if CLI_CAPS.get("outputs"):
                if "html" not in outputs_final:
                    outputs_final.append("html")
            elif not outputs_final:
                # Older CLI ignores outputs; keep a sane display default
                outputs_final = ["json"]
            
            # Parse optional JSON fields
            parsed_backend_kwargs = {}
            try:
                if backend_kwargs_raw.strip():
                    parsed_backend_kwargs.update(json.loads(backend_kwargs_raw))
            except Exception:
                st.error("Invalid JSON in Backend Kwargs")
                st.session_state.benchmark_running = False
                st.stop()

            if api_key:
                parsed_backend_kwargs["api_key"] = api_key
            
            parsed_data_args = None
            if data_args.strip():
                try:
                    parsed_data_args = json.loads(data_args)
                except Exception:
                    st.error("Invalid JSON in Data Args")
                    st.session_state.benchmark_running = False
                    st.stop()
            
            parsed_processor_args = None
            if processor_args.strip():
                try:
                    parsed_processor_args = json.loads(processor_args)
                except Exception:
                    st.error("Invalid JSON in Processor Args")
                    st.session_state.benchmark_running = False
                    st.stop()
            
            parsed_output_extras = None
            if output_extras.strip():
                try:
                    parsed_output_extras = json.loads(output_extras)
                except Exception:
                    st.error("Invalid JSON in Output Extras")
                    st.session_state.benchmark_running = False
                    st.stop()
            
            # Build command
            cmd = [
                "guidellm", "benchmark",
                "--target", target,
                "--max-seconds", str(max_seconds),
                "--max-requests", str(max_requests),
                "--data", data_config if isinstance(data_config, str) else json.dumps(data_config)
            ]

            # Profile / rate-type (compat across GuideLLM CLI versions)
            if CLI_CAPS.get("profile"):
                cmd.extend(["--profile", profile])
            elif CLI_CAPS.get("rate_type"):
                cmd.extend(["--rate-type", profile])
            else:
                cmd.extend(["--profile", profile])

            # Output path flags (compat across GuideLLM CLI versions)
            if CLI_CAPS.get("output_dir"):
                cmd.extend(["--output-dir", str(output_dir)])
            elif CLI_CAPS.get("output_path"):
                cmd.extend(["--output-path", str(output_dir)])
            else:
                cmd.extend(["--output-dir", str(output_dir)])

            if CLI_CAPS.get("outputs"):
                cmd.extend(["--outputs", ",".join(outputs_final)])
            
            if effective_model:
                cmd.extend(["--model", effective_model])
            
            # Add processor (always prefer explicit processor; fallback to Model Name)
            if processor_effective:
                cmd.extend(["--processor", processor_effective])
            if parsed_processor_args is not None:
                cmd.extend(["--processor-args", json.dumps(parsed_processor_args)])
            
            # Add rate if specified
            if rate:
                cmd.extend(["--rate", str(rate)])

            if rampup is not None and rampup > 0:
                if CLI_CAPS.get("rampup"):
                    cmd.extend(["--rampup", str(rampup)])
                else:
                    st.warning("`--rampup` is not supported by this GuideLLM CLI version; ignoring.")

            if data_sampler != "none":
                cmd.extend(["--data-sampler", data_sampler])

            if parsed_data_args is not None:
                cmd.extend(["--data-args", json.dumps(parsed_data_args)])

            if parsed_backend_kwargs:
                if CLI_CAPS.get("backend_kwargs"):
                    cmd.extend(["--backend-kwargs", json.dumps(parsed_backend_kwargs)])
                elif CLI_CAPS.get("backend_args"):
                    cmd.extend(["--backend-args", json.dumps(parsed_backend_kwargs)])
                else:
                    cmd.extend(["--backend-kwargs", json.dumps(parsed_backend_kwargs)])

            if parsed_output_extras is not None:
                if CLI_CAPS.get("output_extras"):
                    cmd.extend(["--output-extras", json.dumps(parsed_output_extras)])
                else:
                    st.warning("`--output-extras` is not supported by this GuideLLM CLI version; ignoring.")

            if sample_requests and sample_requests > 0:
                if CLI_CAPS.get("sample_requests"):
                    cmd.extend(["--sample-requests", str(sample_requests)])
                else:
                    st.warning("`--sample-requests` is not supported by this GuideLLM CLI version; ignoring.")

            if disable_progress:
                cmd.append("--disable-progress")

            if disable_console_outputs:
                cmd.append("--disable-console-outputs")

            if detect_saturation:
                if CLI_CAPS.get("detect_saturation"):
                    cmd.append("--detect-saturation")
                else:
                    st.warning("`--detect-saturation` is not supported by this GuideLLM CLI version; ignoring.")
            
            # Set environment variables
            env = os.environ.copy()
            if hf_token:
                env["HUGGING_FACE_HUB_TOKEN"] = hf_token
            env["GUIDELLM__MAX_CONCURRENCY"] = str(max_concurrency)
            if CLI_CAPS.get("outputs") and "html" in outputs_final and not env.get("GUIDELLM__REPORT_GENERATION__SOURCE"):
                report_url = "https://blog.vllm.ai/guidellm/ui/latest"
                local_template = _ensure_local_report_template(report_url)
                if local_template:
                    env["GUIDELLM__REPORT_GENERATION__SOURCE"] = str(local_template)
                else:
                    env["GUIDELLM__REPORT_GENERATION__SOURCE"] = report_url
            
            # Display command being executed (properly quoted for shell)
            import shlex
            display_cmd = " ".join(shlex.quote(arg) for arg in cmd)
            st.code(display_cmd, language="bash")
            
            # Create placeholder for output
            output_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Execute command
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env,
                    bufsize=1
                )
                
                output_lines = []
                benchmark_stats = []
                stats_started = False
                start_time = time.time()
                
                # Create placeholders for real-time display
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Real-time output display
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line_stripped = line.strip()
                        output_lines.append(line_stripped)
                        current_time = time.time()
                        elapsed = current_time - start_time
                        progress = min(elapsed / max_seconds, 1.0)
                        
                        progress_bar.progress(progress)
                        status_text.text(f"Running... ({elapsed:.0f}s elapsed)")
                        
                        # Parse real-time benchmark progress and store in session state
                        if "│" in line_stripped and ("req/s" in line_stripped or "Lat" in line_stripped):
                            # Extract live metrics from the progress box
                            try:
                                # Parse lines like: │ [00:47:39] ⠦ 100% synchronous (complete) Req: 0.3 req/s, 3.88s Lat, 1.0 Conc, 14 Comp, 1 Inc, 0 Err │
                                if "req/s" in line_stripped and "Lat" in line_stripped:
                                    import re
                                    
                                    # Extract metrics using regex
                                    req_match = re.search(r'Req:\s*([\d.]+)\s*req/s', line_stripped)
                                    lat_match = re.search(r'([\d.]+)s\s*Lat', line_stripped)
                                    conc_match = re.search(r'([\d.]+)\s*Conc', line_stripped)
                                    comp_match = re.search(r'(\d+)\s*Comp', line_stripped)
                                    
                                    tok_match = re.search(r'Tok:\s*([\d.]+)\s*gen/s,\s*([\d.]+)\s*tot/s', line_stripped)
                                    ttft_match = re.search(r'([\d.]+)ms\s*TTFT', line_stripped)
                                    
                                    if req_match and lat_match and tok_match and ttft_match:
                                        # Store live metrics in session state for sidebar display
                                        st.session_state.live_metrics = {
                                            "requests_per_sec": req_match.group(1),
                                            "latency": f"{lat_match.group(1)}s",
                                            "concurrency": conc_match.group(1) if conc_match else "1",
                                            "completed": comp_match.group(1) if comp_match else "0",
                                            "gen_tokens_per_sec": tok_match.group(1),
                                            "total_tokens_per_sec": tok_match.group(2),
                                            "ttft": f"{ttft_match.group(1)}ms"
                                        }
                            except:
                                pass  # If parsing fails, continue
                        
                        # Check for progress bar updates
                        elif "Generating..." in line_stripped and "━" in line_stripped:
                            # Show progress info
                            with progress_placeholder.container():
                                st.info(f"🔄 {line_stripped}")
                        
                        # Check for phase updates
                        elif any(phase in line_stripped for phase in ["Creating backend", "Creating request loader", "Created loader"]):
                            with status_placeholder.container():
                                st.info(f"📋 {line_stripped}")
                        
                        # Check for benchmark stats table - be more flexible with patterns
                        elif any(pattern in line_stripped for pattern in ["Benchmark Stats:", "===============", "Rate Type", "|synchronous|", "|throughput|", "|concurrent|", "|constant|", "|poisson|", "|sweep|"]):
                            stats_started = True
                        
                        if stats_started:
                            benchmark_stats.append(line_stripped)
                        
                        # Display last 10 lines of output (reduced to make room for live metrics)
                        recent_output = "\n".join(output_lines[-10:])
                        output_placeholder.text_area(
                            "Console Output",
                            value=recent_output,
                            height=200,
                            key=f"output_{len(output_lines)}"
                        )
                
                process.wait()
                
                if process.returncode == 0:
                    st.success("✅ Benchmark completed successfully!")
                    
                    # Store final results in session state for sidebar display
                    if benchmark_stats:
                        final_stats = "\n".join(benchmark_stats)
                        st.session_state.final_benchmark_results = final_stats
                    else:
                        # Try to parse from all output lines
                        all_output = "\n".join(output_lines)
                        st.session_state.final_benchmark_results = all_output
                    
                    # Load and display results
                    output_files = {
                        "html": _find_latest_file(output_dir, ["*.html"]),
                        "json": _find_latest_file(output_dir, ["*.json"]),
                        "yaml": _find_latest_file(output_dir, ["*.yaml", "*.yml"]),
                        "csv": _find_latest_file(output_dir, ["*.csv"])
                    }
                    
                    results = None
                    if output_files["json"] and output_files["json"].exists():
                        try:
                            with open(output_files["json"], "r") as f:
                                results = json.load(f)
                        except Exception:
                            results = None
                    elif output_files["yaml"] and output_files["yaml"].exists():
                        try:
                            with open(output_files["yaml"], "r") as f:
                                results = yaml.safe_load(f)
                        except Exception:
                            results = None
                    
                    html_report = None
                    if output_files["html"] and output_files["html"].exists():
                        try:
                            html_report = output_files["html"].read_text(encoding="utf-8")
                        except Exception:
                            html_report = None

                    compact_report_path = None
                    compact_html_report = None
                    if results:
                        theme_base = st.get_option("theme.base") or "dark"
                        if theme_base not in ("light", "dark"):
                            theme_base = "dark"
                        compact_report_path = _render_compact_report(
                            results, output_dir, theme=theme_base
                        )
                        if compact_report_path and Path(compact_report_path).exists():
                            try:
                                compact_html_report = Path(compact_report_path).read_text(encoding="utf-8")
                            except Exception:
                                compact_html_report = None
                    
                    # Store in session state with benchmark stats
                    final_stats = "\n".join(benchmark_stats) if benchmark_stats else None
                    result_entry = {
                        "timestamp": timestamp,
                        "model": effective_model if effective_model else model_name,
                        "target": target,
                        "config": config_display,
                        "results": results,
                        "benchmark_stats": final_stats,
                        "output_dir": str(output_dir),
                        "output_files": {k: str(v) if v else None for k, v in output_files.items()},
                        "html_report": html_report,
                        "compact_report_path": str(compact_report_path) if compact_report_path else None,
                        "compact_html_report": compact_html_report
                    }
                    st.session_state.results_history.append(result_entry)
                    
                else:
                    st.error(f"❌ Benchmark failed with return code {process.returncode}")
                    all_output = "\n".join(output_lines)
                    if "Invalid rates in sweep; aborting" in all_output:
                        st.warning(
                            "Sweep failed because the throughput phase produced no successful requests. "
                            "Try increasing max-seconds/max-requests, lowering rate/rampup, or limiting "
                            "max concurrency, then rerun."
                        )
                    
            except Exception as e:
                st.error(f"❌ Error running benchmark: {str(e)}")
            
            finally:
                st.session_state.benchmark_running = False
                # Only clear progress elements, keep results visible
                try:
                    progress_bar.empty()
                    status_text.empty()
                    # Live metrics now stored in session state for sidebar
                    # DON'T clear output_placeholder - it shows console history
                    progress_placeholder.empty() 
                    status_placeholder.empty()
                except:
                    pass

    # Show live metrics below the button (main area)
    if hasattr(st.session_state, 'live_metrics') and st.session_state.benchmark_running:
        st.subheader("🔥 Live Performance Metrics")
        metrics = st.session_state.live_metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚀 Requests/sec", metrics["requests_per_sec"])
        with col2:
            st.metric("⚡ Tokens/sec", metrics["gen_tokens_per_sec"])
        with col3:
            st.metric("⏱️ Latency", metrics["latency"])
        with col4:
            st.metric("🎯 TTFT", metrics["ttft"])
    
    # Show final results table in main area
    elif hasattr(st.session_state, 'final_benchmark_results'):
        st.success("🎉 Benchmark completed! Check the sidebar for key metrics.")
        # Show complete results table in main area (prefer JSON results)
        latest_result = st.session_state.results_history[-1] if st.session_state.results_history else None
        if latest_result and latest_result.get("results"):
            compact = _extract_compact_metrics(latest_result["results"])
            if compact:
                st.subheader("📊 Complete Results")
                results_data = {
                    "Metric": [name for name, _ in compact["rows"]],
                    "Value": [value for _, value in compact["rows"]],
                }
                df = pd.DataFrame(results_data)
                st.dataframe(df, width="stretch", hide_index=True)
        else:
            # Fallback to console parsing
            final_stats = st.session_state.final_benchmark_results
            try:
                for line in final_stats.split('\n'):
                    if any(k in line for k in ["synchronous", "throughput", "concurrent", "constant", "poisson", "sweep"]):
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 10:
                            st.subheader("📊 Complete Results")
                            results_data = {
                                "Metric": ["Profile", "Requests/Second", "Concurrency", "Output Tokens/sec", "Total Tokens/sec", 
                                          "Mean Latency (ms)", "Median Latency (ms)", "P99 Latency (ms)", 
                                          "Mean TTFT (ms)", "Median TTFT (ms)", "P99 TTFT (ms)"],
                                "Value": [parts[0], parts[1], parts[2], parts[3], parts[4], 
                                         parts[5], parts[6], parts[7], parts[8], parts[9], parts[10]]
                            }
                            df = pd.DataFrame(results_data)
                            st.dataframe(df, width="stretch", hide_index=True)
                            break
            except:
                pass

        # Render HTML report inline (latest run)
        if st.session_state.results_history and show_html_report:
            latest_result = st.session_state.results_history[-1]
            html_report = latest_result.get("compact_html_report") or latest_result.get("html_report")
            if html_report:
                st.subheader("🧾 HTML Report")
                st.components.v1.html(html_report, height=800, scrolling=True)

with col2:
    st.header("📊 Quick Stats")
    
    # Show final results if just completed
    if hasattr(st.session_state, 'final_benchmark_results'):
        # Prefer JSON results for accuracy
        parsed_successfully = False
        latest_result = st.session_state.results_history[-1] if st.session_state.results_history else None
        if latest_result and latest_result.get("results"):
            quick = _extract_quick_stats(latest_result["results"])
            if quick:
                st.subheader("✅ Final Results")
                st.metric("🚀 Requests/sec", quick.get("requests_per_sec", "N/A"))
                st.metric("⚡ Tokens/sec", quick.get("tokens_per_sec", "N/A"))
                st.metric("⏱️ Latency", f"{quick.get('latency_ms', 'N/A')} ms")
                st.metric("🎯 TTFT", f"{quick.get('ttft_ms', 'N/A')} ms")
                parsed_successfully = True
        
        if not parsed_successfully:
            final_stats = st.session_state.final_benchmark_results
            # Parse and show key metrics from console output
            try:
                for line in final_stats.split('\n'):
                    if any(k in line for k in ["synchronous", "throughput", "concurrent", "constant", "poisson", "sweep"]):
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 10:
                            st.subheader("✅ Final Results")
                            st.metric("🚀 Requests/sec", parts[1])
                            st.metric("⚡ Tokens/sec", parts[3])
                            st.metric("⏱️ Latency", f"{parts[5]} ms")
                            st.metric("🎯 TTFT", f"{parts[8]} ms")
                            parsed_successfully = True
                            break
            except Exception:
                pass  # Silent error handling
        
        # Fallback: show basic info if parsing failed
        if not parsed_successfully:
            st.subheader("✅ Benchmark Completed")
            st.info("Results available in history below")
            # Try to show some basic stats from YAML results if available
            if st.session_state.results_history:
                latest = st.session_state.results_history[-1]
                if "results" in latest and latest["results"]:
                    results = latest["results"]
                    if "summary" in results:
                        summary = results["summary"]
                        st.metric("🚀 Throughput", f"{summary.get('throughput', 'N/A')}")
                        st.metric("⏱️ Mean Latency", f"{summary.get('mean_latency', 'N/A')}")
                        parsed_successfully = True
    
    # Always show general stats and historical data when available
    elif st.session_state.results_history:
        total_runs = len(st.session_state.results_history)
        latest_run = st.session_state.results_history[-1]
        
        st.metric("Total Runs", total_runs)
        st.metric("Latest Model", latest_run["model"])
        st.metric("Latest Timestamp", latest_run["timestamp"])
        
        # Beautiful latest results display
        if latest_run.get("results"):
            quick = _extract_quick_stats(latest_run["results"])
            if quick:
                st.subheader("🏆 Latest Performance")
                st.metric("🚀 Requests/sec", quick.get("requests_per_sec", "N/A"))
                st.metric("⚡ Tokens/sec", quick.get("tokens_per_sec", "N/A"))
                st.metric("⏱️ Latency", f"{quick.get('latency_ms', 'N/A')} ms")
                st.metric("🎯 TTFT", f"{quick.get('ttft_ms', 'N/A')} ms")
        elif "benchmark_stats" in latest_run and latest_run["benchmark_stats"]:
            st.subheader("🏆 Latest Performance")
            
            # Parse the latest benchmark stats for display
            try:
                stats_lines = latest_run["benchmark_stats"].split('\n')
                for line in stats_lines:
                    if any(k in line for k in ["synchronous", "throughput", "concurrent", "constant", "poisson", "sweep"]):
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 10:
                            st.metric("🚀 Requests/sec", parts[1])
                            st.metric("⚡ Tokens/sec", parts[3])
                            st.metric("⏱️ Latency", f"{parts[5]} ms")
                            st.metric("🎯 TTFT", f"{parts[8]} ms")
                            break
            except:
                pass
        
        # Fallback to old results format
        elif "results" in latest_run and latest_run["results"]:
            results = latest_run["results"]
            if "summary" in results:
                summary = results["summary"]
                if "throughput" in summary:
                    st.metric("Throughput (req/s)", f"{summary['throughput']:.2f}")
                if "mean_latency" in summary:
                    st.metric("Mean Latency (ms)", f"{summary['mean_latency']:.2f}")
    else:
        st.info("No benchmark runs yet")

# Results section
if st.session_state.results_history:
    st.header("📈 Results History")
    
    # Results table
    results_data = []
    for result in st.session_state.results_history:
        row = {
            "Timestamp": result["timestamp"],
            "Model": result["model"],
            "Target": result["target"][:50] + "..." if len(result["target"]) > 50 else result["target"],
            "Profile": result["config"]["Profile"],
            "Duration": result["config"]["Max Duration"],
            "Formats": ", ".join([k for k, v in (result.get("output_files") or {}).items() if v]) or "N/A"
        }
        
        # Add performance metrics if available
        if "results" in result and result["results"] and "summary" in result["results"]:
            summary = result["results"]["summary"]
            row["Throughput"] = f"{summary.get('throughput', 0):.2f} req/s"
            row["Mean Latency"] = f"{summary.get('mean_latency', 0):.2f} ms"
        
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, width="stretch")
    
    # Detailed results viewer
    st.subheader("🔍 Detailed Results")
    
    selected_run = st.selectbox(
        "Select run to view details",
        options=range(len(st.session_state.results_history)),
        format_func=lambda x: f"{st.session_state.results_history[x]['timestamp']} - {st.session_state.results_history[x]['model']}"
    )
    
    if selected_run is not None:
        result = st.session_state.results_history[selected_run]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration")
            st.json(result["config"])
        
        with col2:
            st.subheader("Results")
            
            # Show benchmark stats if available
            if "benchmark_stats" in result and result["benchmark_stats"]:
                st.subheader("📊 Benchmark Stats")
                
                # Parse and display beautifully
                try:
                    stats_lines = result["benchmark_stats"].split('\n')
                    data_row = None
                    for line in stats_lines:
                        if any(k in line for k in ["synchronous", "throughput", "concurrent", "constant", "poisson", "sweep"]):
                            data_row = line
                            break
                    
                    if data_row:
                        parts = [p.strip() for p in data_row.split("|")]
                        if len(parts) >= 10:
                            # Key metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🚀 Requests/sec", parts[1])
                                st.metric("⏱️ Mean Latency", f"{parts[5]} ms")
                            with col2:
                                st.metric("⚡ Tokens/sec", parts[3])
                                st.metric("🎯 TTFT", f"{parts[8]} ms")
                            
                            # Full details in expander
                            with st.expander("📋 Full Benchmark Stats"):
                                st.code(result["benchmark_stats"], language="text")
                        else:
                            st.code(result["benchmark_stats"], language="text")
                    else:
                        st.code(result["benchmark_stats"], language="text")
                except:
                    st.code(result["benchmark_stats"], language="text")
            
            # Show detailed results
            if "results" in result and result["results"]:
                with st.expander("📋 Detailed Results"):
                    st.json(result["results"])
            else:
                st.info("No detailed results available")

            # Inline HTML report for selected run
            if show_html_report and (result.get("compact_html_report") or result.get("html_report")):
                st.subheader("🧾 HTML Report")
                html_content = result.get("compact_html_report") or result.get("html_report")
                st.components.v1.html(html_content, height=800, scrolling=True)
        
        # Download button for results
        output_files = result.get("output_files") or {}
        if result.get("compact_report_path"):
            output_files = dict(output_files)
            output_files["compact"] = result["compact_report_path"]
        available_files = {k: v for k, v in output_files.items() if v and Path(v).exists()}
        if available_files:
            selected_format = st.selectbox(
                "Download Format",
                options=list(available_files.keys()),
                index=0
            )
            download_path = Path(available_files[selected_format])
            results_content = download_path.read_text(encoding="utf-8")
            mime_type = "text/html" if selected_format == "html" else "text/plain"
            st.download_button(
                label=f"📥 Download Results ({selected_format.upper()})",
                data=results_content,
                file_name=f"benchmark-{result['timestamp']}.{selected_format}",
                mime=mime_type
            )

# Footer
st.markdown("---")
st.markdown("PoC App by <a href='http://red.ht/cai-team' target='_blank'>red.ht/cai-team</a>&nbsp;&nbsp;-&nbsp;&nbsp;*Built with ❤️ using Streamlit and GuideLLM*", unsafe_allow_html=True)
