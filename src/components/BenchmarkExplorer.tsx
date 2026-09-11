"use client";

import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { withBasePath } from "@/lib/basePath";

type Platform = "a6000" | "orin";
type PowerMode = "maxn" | "20w" | "15w" | "10w";
type Precision = "fp32" | "amp" | "trt_fp32" | "trt_fp16";
type Family = "relative" | "metric";
type Dataset = "NYUD" | "KITTI";
type AccuracyMode = "tf32" | "trt_fp32" | "trt_fp16";
type Encoder = "S" | "B" | "L" | "MNV4-S" | "MNV4-M" | "MNV4-M-SLIM-SPF";

type BenchmarkRow = {
  family: Family;
  encoder: Encoder;
  resolution: number;
  mode: Precision;
  latency: number;
  fps: number;
};

type AccuracyRow = {
  family: Family;
  encoder: Encoder;
  resolution: number;
  dataset: Dataset;
  mode: AccuracyMode;
  delta1: number;
  absRel: number;
  rmse: number;
};

type StrictRow = {
  method: string;
  modelScale: string;
  variant: string;
  paramsM: number;
  dataset: Dataset;
  delta1: number;
  affineDelta1: number;
  speedInput: string;
  latency: number;
  fps: number;
};

type ChartDatum = {
  id: string;
  methodFamily: string;
  label: string;
  pointLabel: string;
  resolutionLabel: string;
  resolution?: number;
  encoder?: Encoder;
  latency: number;
  fps: number;
  accuracy: number;
  paramsM?: number;
  absRel?: number;
  rmse?: number;
  color: string;
  isDepthART: boolean;
};

type PlotPoint = ChartDatum & { x: number; y: number };

const modelColors: Record<Encoder, string> = {
  S: "#ff746a",
  B: "#8d8bff",
  L: "#4fc5ff",
  "MNV4-S": "#54d69b",
  "MNV4-M": "#ffc857",
  "MNV4-M-SLIM-SPF": "#f58bd7"
};

const encoderLabels: Record<Encoder, string> = {
  S: "TinyViM-S",
  B: "TinyViM-B",
  L: "TinyViM-L",
  "MNV4-S": "MobileNetV4-S",
  "MNV4-M": "MobileNetV4-M",
  "MNV4-M-SLIM-SPF": "MobileNetV4-M-slim-SPF"
};

const encoderOrder = Object.keys(encoderLabels) as Encoder[];

const baselineColors: Record<string, string> = {
  MiDaS: "#9aa9bd",
  "Depth-Anything": "#78a9ff",
  "Depth-Anything-V2": "#4777ff",
  "UniDepth-V2": "#59d5a4",
  ZoeDepth: "#ffc75f",
  "MoGe-v2": "#45d5d0",
  "Metric3D-v2": "#c68cff",
  DepthPro: "#ff8eba",
  "YOLO26-Depth": "#f5a45d",
  YOLO26: "#f5a45d",
  ZipDepth: "#e9dd6c"
};

const precisionOptions: { value: Precision; label: string }[] = [
  { value: "fp32", label: "PyTorch FP32" },
  { value: "amp", label: "PyTorch AMP" },
  { value: "trt_fp32", label: "TensorRT FP32" },
  { value: "trt_fp16", label: "TensorRT FP16" }
];

const powerOptions: { value: PowerMode; label: string }[] = [
  { value: "maxn", label: "MAXN" },
  { value: "20w", label: "20W" },
  { value: "15w", label: "15W" },
  { value: "10w", label: "10W" }
];

const chart = { width: 920, height: 380, left: 74, right: 28, top: 34, bottom: 58 };

function parseCsvLine(line: string) {
  const fields: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        field += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      fields.push(field);
      field = "";
    } else {
      field += char;
    }
  }
  fields.push(field);
  return fields;
}

function csvRecords(csv: string) {
  const [headerLine, ...lines] = csv.trim().split(/\r?\n/);
  const headers = parseCsvLine(headerLine);
  return lines.filter(Boolean).map((line) => {
    const fields = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, fields[index] ?? ""]));
  });
}

function parseCsv(csv: string): BenchmarkRow[] {
  return csvRecords(csv).map((row) => ({
    family: row.family as Family,
    encoder: row.encoder as Encoder,
    resolution: Number(row.resolution),
    mode: row.mode as Precision,
    latency: Number(row.model_mean_ms),
    fps: Number(row.model_fps)
  }));
}

function parseAccuracyCsv(csv: string): AccuracyRow[] {
  return csvRecords(csv).map((row) => ({
    family: row.family as Family,
    encoder: row.encoder as Encoder,
    resolution: Number(row.nominal_resolution),
    dataset: row.dataset as Dataset,
    mode: row.mode as AccuracyMode,
    delta1: Number(row.delta1),
    absRel: Number(row.abs_rel),
    rmse: Number(row.rmse)
  }));
}

function parseStrictCsv(csv: string): StrictRow[] {
  return csvRecords(csv).map((row) => ({
    method: row.method,
    modelScale: row.model_scale,
    variant: row.variant,
    paramsM: Number(row.params_m),
    dataset: row.dataset as Dataset,
    delta1: Number(row.delta1),
    affineDelta1: Number(row.affine_delta1),
    speedInput: row.actual_speed_input,
    latency: Number(row.model_only_latency_ms),
    fps: Number(row.model_only_fps)
  }));
}

function accuracyMode(precision: Precision): AccuracyMode {
  return precision === "trt_fp32" || precision === "trt_fp16" ? precision : "tf32";
}

function csvFile(platform: Platform, power: PowerMode, precision: Precision) {
  if (platform === "a6000") return "summary_A6000.csv";
  if (power === "maxn" && precision === "fp32") return "summary_maxn_pytorch_fp32_depthart.csv";
  return `summary_${power}.csv`;
}

function supports(platform: Platform, power: PowerMode, precision: Precision) {
  if (platform === "a6000") return true;
  if (precision === "amp") return false;
  if (precision === "fp32") return power === "maxn";
  return true;
}

function geometricTicks(min: number, max: number) {
  if (min === max) return [min];
  const low = min * 0.82;
  const high = max * 1.22;
  return Array.from({ length: 5 }, (_, i) => low * Math.pow(high / low, i / 4));
}

function formatLatency(value: number) {
  if (value < 10) return value.toFixed(2);
  if (value < 100) return value.toFixed(1);
  return value.toFixed(0);
}

function strictLabel(row: StrictRow) {
  if (row.method === "DepthART" || row.method === "DepthART-Metric") return `${row.method}-${row.modelScale}`;
  if (row.modelScale === "default") return row.method;
  return `${row.method} ${row.modelScale}`;
}

function fisheye(value: number, focus: number | null, distortion = 1.7) {
  if (focus === null || value === focus) return value;
  const left = value < focus;
  const side = left ? focus : 1 - focus;
  const distance = Math.abs(value - focus);
  if (side <= 0 || distance <= 0) return value;
  const magnified = side * (distortion + 1) / (distortion + side / distance);
  return focus + (left ? -magnified : magnified);
}

function latencyPosition(value: number, range: { min: number; max: number }, focus: number | null) {
  const low = range.min * 0.82;
  const high = range.max * 1.22;
  const normalized = Math.log(value / low) / Math.log(high / low);
  return chart.left + fisheye(normalized, focus) * (chart.width - chart.left - chart.right);
}

function latencyAtFocus(range: { min: number; max: number }, focus: number) {
  const low = range.min * 0.82;
  const high = range.max * 1.22;
  return low * Math.pow(high / low, focus);
}

function strictAnnotation(point: PlotPoint) {
  const key = `${point.encoder}-${point.resolution}`;
  const positions: Record<string, { x: number; y: number; anchor: "start" | "end" }> = {
    "S-224": { x: -8, y: -42, anchor: "end" },
    "B-224": { x: -4, y: -57, anchor: "end" },
    "S-448": { x: -5, y: -72, anchor: "end" },
    "B-448": { x: 10, y: -30, anchor: "start" },
    "L-224": { x: 12, y: -60, anchor: "start" },
    "L-448": { x: 14, y: -62, anchor: "start" }
  };
  return positions[key] ?? { x: 12, y: -34, anchor: "start" as const };
}

export default function BenchmarkExplorer() {
  const [platform, setPlatform] = useState<Platform>("a6000");
  const [power, setPower] = useState<PowerMode>("maxn");
  const [precision, setPrecision] = useState<Precision>("fp32");
  const [family, setFamily] = useState<Family>("relative");
  const [dataset, setDataset] = useState<Dataset>("NYUD");
  const [rows, setRows] = useState<BenchmarkRow[]>([]);
  const [accuracyRows, setAccuracyRows] = useState<AccuracyRow[]>([]);
  const [strictRows, setStrictRows] = useState<StrictRow[]>([]);
  const [active, setActive] = useState<string>("DepthART-S-S_224-NYUD");
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [lensFocus, setLensFocus] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const lensFrame = useRef<number | null>(null);
  const useStrict = platform === "a6000" && precision === "fp32";

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch(withBasePath("/benchmarks/DepthART_A6000_metrics.csv")),
      fetch(withBasePath("/benchmarks/monocular_depth_nyud_kitti_strict_fp32.csv"))
    ])
      .then(async ([accuracyResponse, strictResponse]) => {
        if (!accuracyResponse.ok || !strictResponse.ok) throw new Error("Accuracy benchmark CSV could not be loaded");
        return Promise.all([accuracyResponse.text(), strictResponse.text()]);
      })
      .then(([accuracyText, strictText]) => {
        if (cancelled) return;
        setAccuracyRows(parseAccuracyCsv(accuracyText));
        setStrictRows(parseStrictCsv(strictText));
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    fetch(withBasePath(`/benchmarks/${csvFile(platform, power, precision)}`))
      .then((response) => {
        if (!response.ok) throw new Error(`Benchmark CSV returned ${response.status}`);
        return response.text();
      })
      .then((text) => {
        if (cancelled) return;
        setRows(parseCsv(text));
        setStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });
    return () => { cancelled = true; };
  }, [platform, power, precision]);

  const matched = useMemo(
    () => rows
      .filter((row) => row.family === family && row.mode === precision)
      .flatMap((row) => {
        const metrics = accuracyRows.find((item) =>
          item.family === family && item.encoder === row.encoder && item.resolution === row.resolution &&
          item.dataset === dataset && item.mode === accuracyMode(precision)
        );
        return metrics ? [{ ...row, metrics }] : [];
      }),
    [rows, accuracyRows, family, dataset, precision]
  );

  const chartData = useMemo<ChartDatum[]>(() => {
    if (useStrict) {
      return strictRows
        .filter((row) => row.dataset === dataset)
        .filter((row) => family === "metric" ? row.method === "DepthART-Metric" : row.method !== "DepthART-Metric")
        .map((row) => {
          const isDepthART = row.method === "DepthART" || row.method === "DepthART-Metric";
          const encoder = isDepthART && encoderOrder.includes(row.modelScale as Encoder) ? row.modelScale as Encoder : undefined;
          const resolution = Number(row.speedInput.match(/\d+/)?.[0]);
          return {
            id: `${row.method}-${row.modelScale}-${row.variant}-${row.dataset}`,
            methodFamily: row.method,
            label: strictLabel(row),
            pointLabel: isDepthART ? `${row.modelScale} · ${resolution}` : strictLabel(row),
            resolutionLabel: row.speedInput,
            resolution,
            encoder,
            latency: row.latency,
            fps: row.fps,
            accuracy: family === "metric" ? row.delta1 : row.affineDelta1,
            paramsM: row.paramsM,
            color: encoder ? modelColors[encoder] : (baselineColors[row.method] ?? "#a9b7c9"),
            isDepthART
          };
        });
    }
    return matched.map((row) => {
      const method = family === "metric" ? "DepthART-Metric" : "DepthART";
      const paramsM = strictRows.find((item) =>
        item.method === method && item.modelScale === row.encoder && item.variant === `${row.encoder}_${row.resolution}`
      )?.paramsM;
      return {
        id: `${row.encoder}-${row.resolution}`,
        methodFamily: method,
        label: `DepthART-${row.encoder}`,
        pointLabel: `${row.encoder} · ${row.resolution}`,
        resolutionLabel: `${row.resolution} × ${row.resolution}`,
        resolution: row.resolution,
        encoder: row.encoder,
        latency: row.latency,
        fps: row.fps,
        accuracy: row.metrics.delta1,
        paramsM,
        absRel: row.metrics.absRel,
        rmse: row.metrics.rmse,
        color: modelColors[row.encoder],
        isDepthART: true
      };
    });
  }, [useStrict, strictRows, dataset, family, matched]);

  const latencyRange = useMemo(() => {
    const values = chartData.map((row) => row.latency);
    return values.length ? { min: Math.min(...values), max: Math.max(...values) } : { min: 1, max: 10 };
  }, [chartData]);

  const accuracyRange = useMemo(() => {
    if (!chartData.length) return { min: 0.9, max: 1 };
    const values = chartData.map((row) => row.accuracy);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const padding = Math.max((rawMax - rawMin) * 0.2, 0.003);
    return { min: Math.max(0, rawMin - padding), max: Math.min(1, rawMax + padding) };
  }, [chartData]);

  const yTicks = useMemo(
    () => Array.from({ length: 5 }, (_, index) => accuracyRange.min + (accuracyRange.max - accuracyRange.min) * index / 4),
    [accuracyRange]
  );
  const xTicks = useMemo(() => chartData.length ? geometricTicks(latencyRange.min, latencyRange.max) : [], [chartData.length, latencyRange]);

  const plotPoints = useMemo<PlotPoint[]>(() => {
    if (!chartData.length) return [];
    const plotHeight = chart.height - chart.top - chart.bottom;
    const yPosition = (value: number) => chart.top + ((accuracyRange.max - value) / (accuracyRange.max - accuracyRange.min)) * plotHeight;
    return chartData.map((row) => ({ ...row, x: latencyPosition(row.latency, latencyRange, lensFocus), y: yPosition(row.accuracy) }));
  }, [chartData, latencyRange, accuracyRange, lensFocus]);

  const activePoint = plotPoints.find((point) => point.id === active) ?? plotPoints[0];
  const activeFamilyPoints = useMemo(
    () => activePoint
      ? plotPoints.filter((point) => point.methodFamily === activePoint.methodFamily).sort((a, b) => a.latency - b.latency)
      : [],
    [plotPoints, activePoint]
  );
  const platformLabel = platform === "a6000" ? "RTX A6000" : `Jetson Orin NX · ${power.toUpperCase()}`;
  const precisionLabel = precisionOptions.find((option) => option.value === precision)?.label;
  const datasetLabel = dataset === "NYUD" ? "NYUD v2" : "KITTI";
  const familyLabel = family === "relative" ? "Relative" : "Metric";
  const loading = status === "loading" || (useStrict ? !strictRows.length : !accuracyRows.length);
  const lensX = lensFocus === null ? null : chart.left + lensFocus * (chart.width - chart.left - chart.right);
  const lensLatency = lensFocus === null ? null : latencyAtFocus(latencyRange, lensFocus);
  const additionalComparisonFamilies = [
    {
      label: "YOLO26-Depth",
      className: "family-yolo",
      point: plotPoints.find((point) => point.methodFamily === "YOLO26-Depth")
    },
    {
      label: "ZipDepth",
      className: "family-zipdepth",
      point: plotPoints.find((point) => point.methodFamily === "ZipDepth")
    }
  ];

  function choosePlatform(next: Platform) {
    setPlatform(next);
    if (next === "orin" && !supports("orin", power, precision)) setPrecision("trt_fp16");
  }

  function choosePower(next: PowerMode) {
    setPower(next);
    if (!supports("orin", next, precision)) setPrecision("trt_fp16");
  }

  function moveLens(event: PointerEvent<SVGSVGElement>) {
    const eventTarget = event.target;
    if (eventTarget instanceof Element && eventTarget.closest(".interactive-point")) return;
    const bounds = svgRef.current?.getBoundingClientRect();
    if (!bounds) return;
    const svgX = (event.clientX - bounds.left) / bounds.width * chart.width;
    const next = Math.max(0, Math.min(1, (svgX - chart.left) / (chart.width - chart.left - chart.right)));
    const softened = lensFocus === null ? next : lensFocus + (next - lensFocus) * 0.34;
    if (lensFrame.current !== null) cancelAnimationFrame(lensFrame.current);
    lensFrame.current = requestAnimationFrame(() => setLensFocus(softened));
  }

  function releaseLens() {
    if (lensFrame.current !== null) cancelAnimationFrame(lensFrame.current);
    lensFrame.current = requestAnimationFrame(() => setLensFocus(null));
  }

  return (
    <section className="section-card dark benchmark-section" id="results">
      <div className="benchmark-heading">
        <span className="eyebrow">Multi-platform Deployment</span>
        <h2>Accuracy vs Speed</h2>
      </div>

      <div className="benchmark-controls">
        <fieldset className="control-platform">
          <legend>Platform</legend>
          <div className="control-group">
            <button type="button" className={platform === "a6000" ? "active" : ""} onClick={() => choosePlatform("a6000")} aria-pressed={platform === "a6000"}>RTX A6000</button>
            <button type="button" className={platform === "orin" ? "active" : ""} onClick={() => choosePlatform("orin")} aria-pressed={platform === "orin"}>Jetson Orin NX</button>
            <button type="button" className="coming-soon" disabled title="Benchmark data coming soon"><span>Jetson Nano 4GB</span><small>Soon</small></button>
          </div>
        </fieldset>

        <fieldset className={`control-power ${platform === "orin" ? "" : "control-muted"}`}>
          <legend>Power mode</legend>
          <div className="control-group">
            {powerOptions.map((option) => <button type="button" key={option.value} className={power === option.value && platform === "orin" ? "active" : ""} onClick={() => choosePower(option.value)} disabled={platform !== "orin"} aria-pressed={power === option.value && platform === "orin"}>{option.label}</button>)}
          </div>
        </fieldset>

        <fieldset className="control-precision">
          <legend>Compute precision</legend>
          <div className="control-group">
            {precisionOptions.map((option) => {
              const available = supports(platform, power, option.value);
              return <button type="button" key={option.value} className={precision === option.value ? "active" : ""} onClick={() => setPrecision(option.value)} disabled={!available} title={available ? undefined : "This combination is not available in the supplied CSV files"} aria-pressed={precision === option.value}>{option.label}</button>;
            })}
          </div>
        </fieldset>

        <fieldset className="control-task">
          <legend>Depth task</legend>
          <div className="control-group">
            <button type="button" className={family === "relative" ? "active" : ""} onClick={() => setFamily("relative")} aria-pressed={family === "relative"}>Relative depth</button>
            <button type="button" className={family === "metric" ? "active" : ""} onClick={() => setFamily("metric")} aria-pressed={family === "metric"}>Metric depth</button>
          </div>
        </fieldset>

        <fieldset className="control-dataset">
          <legend>Dataset</legend>
          <div className="control-group">
            <button type="button" className={dataset === "NYUD" ? "active" : ""} onClick={() => setDataset("NYUD")} aria-pressed={dataset === "NYUD"}>NYUD v2</button>
            <button type="button" className={dataset === "KITTI" ? "active" : ""} onClick={() => setDataset("KITTI")} aria-pressed={dataset === "KITTI"}>KITTI</button>
          </div>
        </fieldset>

        <div className="benchmark-summary" aria-live="polite">
          <strong>{platformLabel}</strong>
          <span>{precisionLabel}</span>
          <span>{familyLabel} · {datasetLabel}</span>
        </div>
      </div>

      <div className="benchmark-body">
        <div className="chart-panel">
          {status === "error" && <div className="benchmark-message">The benchmark CSV could not be loaded.</div>}
          {loading && status !== "error" && <div className="benchmark-message">Loading benchmark data…</div>}
          <svg ref={svgRef} className={`interactive-chart ${lensFocus === null ? "" : "lens-active"}`} viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label={`${familyLabel} depth accuracy on ${datasetLabel} and latency on ${platformLabel} using ${precisionLabel}`} onPointerMove={moveLens} onPointerLeave={releaseLens}>
            {lensX !== null && lensLatency !== null && <g className="lens-guide" aria-hidden="true">
              <rect className="lens-region" x={lensX - 60} y={chart.top} width="120" height={chart.height - chart.top - chart.bottom} />
              <line x1={lensX} x2={lensX} y1={chart.top} y2={chart.height - chart.bottom} />
              <g className="lens-value" transform={`translate(${lensX} ${chart.height - chart.bottom - 20})`}>
                <rect x="-29" y="0" width="58" height="18" rx="5" />
                <text x="0" y="12.5" textAnchor="middle">{formatLatency(lensLatency)} ms</text>
              </g>
            </g>}
            {yTicks.map((tick) => {
              const y = chart.top + ((accuracyRange.max - tick) / (accuracyRange.max - accuracyRange.min)) * (chart.height - chart.top - chart.bottom);
              return <g key={tick}><line className="chart-grid" x1={chart.left} x2={chart.width - chart.right} y1={y} y2={y} /><text className="chart-tick" x={chart.left - 14} y={y + 4} textAnchor="end">{tick.toFixed(3)}</text></g>;
            })}
            {xTicks.map((tick) => {
              const x = latencyPosition(tick, latencyRange, lensFocus);
              return <g key={tick} className="x-tick" style={{ transform: `translateX(${x}px)` }}><line className="chart-grid vertical" x1="0" x2="0" y1={chart.top} y2={chart.height - chart.bottom} /><text className="chart-tick" x="0" y={chart.height - chart.bottom + 23} textAnchor="middle">{formatLatency(tick)}</text></g>;
            })}
            <line className="chart-axis" x1={chart.left} x2={chart.width - chart.right} y1={chart.height - chart.bottom} y2={chart.height - chart.bottom} />
            <line className="chart-axis" x1={chart.left} x2={chart.left} y1={chart.top} y2={chart.height - chart.bottom} />
            <text className="chart-label axis-title" x={(chart.left + chart.width - chart.right) / 2} y={chart.height - 8} textAnchor="middle">Model latency (ms, log scale) →</text>
            <text className="chart-label axis-title" transform={`translate(19 ${(chart.top + chart.height - chart.bottom) / 2}) rotate(-90)`} textAnchor="middle">{datasetLabel} δ1 ↑</text>

            {activeFamilyPoints.length > 1 && (
              <polyline
                className="family-connector"
                points={activeFamilyPoints.map((point) => `${point.x},${point.y}`).join(" ")}
                stroke={activePoint.color}
              />
            )}

            {plotPoints.map((point) => {
              const selected = activePoint?.id === point.id;
              const familySelected = activePoint?.methodFamily === point.methodFamily;
              const diamond = point.isDepthART && point.resolution === 448;
              const isAddedComparison = point.methodFamily === "YOLO26-Depth" || point.methodFamily === "ZipDepth";
              const isZipDepth = point.methodFamily === "ZipDepth";
              const compact = useStrict;
              const hideDenseDepthARTLabel = useStrict && family === "relative" && point.isDepthART;
              const liftedLabel = useStrict && point.isDepthART && !hideDenseDepthARTLabel;
              const showPointLabel = !isZipDepth && !hideDenseDepthARTLabel && (point.isDepthART || selected);
              const annotation = strictAnnotation(point);
              return (
                <g key={point.id} className={`interactive-point ${isAddedComparison ? "added-comparison" : ""} ${selected ? "selected" : ""} ${familySelected ? "family-highlighted" : "family-muted"}`} style={{ transform: `translate(${point.x}px, ${point.y}px)` }} tabIndex={0} role="button" aria-label={`${point.label}, ${point.resolutionLabel}, ${point.accuracy.toFixed(4)} delta one, ${point.latency.toFixed(3)} milliseconds, ${point.fps.toFixed(1)} FPS`} onMouseEnter={() => setActive(point.id)} onFocus={() => setActive(point.id)} onClick={() => setActive(point.id)}>
                  <circle className="point-hit-target" r={isAddedComparison ? 18 : compact ? 15 : 18} fill="transparent" stroke="transparent" />
                  {liftedLabel && <line className="point-label-leader" x1="0" y1="-8" x2={annotation.x} y2={annotation.y + 6} stroke={point.color} />}
                  {isZipDepth
                    ? <rect className="comparison-marker zipdepth-marker" x={selected ? -7 : -5.5} y={selected ? -7 : -5.5} width={selected ? 14 : 11} height={selected ? 14 : 11} rx="2.5" fill={point.color} stroke="#071d42" strokeWidth="1.5" />
                    : diamond
                    ? <path d={compact ? (selected ? "M 0 -10 L 10 0 L 0 10 L -10 0 Z" : "M 0 -7 L 7 0 L 0 7 L -7 0 Z") : (selected ? "M 0 -14 L 14 0 L 0 14 L -14 0 Z" : "M 0 -10 L 10 0 L 0 10 L -10 0 Z")} fill="#071d42" stroke={point.color} strokeWidth={compact ? "3" : "4"} />
                    : <circle r={compact ? (selected ? 8 : point.isDepthART ? 6 : isAddedComparison ? 5.5 : 4.5) : (selected ? 11 : point.isDepthART ? 8 : 6)} fill={point.color} />}
                  {showPointLabel && <text x={liftedLabel ? annotation.x : 13} y={liftedLabel ? annotation.y : -11} textAnchor={liftedLabel ? annotation.anchor : "start"} fill={point.color}>{point.pointLabel}</text>}
                </g>
              );
            })}
          </svg>
        </div>

        <aside className="benchmark-inspector">
          <span className="inspector-kicker">Selected profile</span>
          {activePoint ? <>
            <h3>{activePoint.label} <em>{activePoint.resolutionLabel}</em></h3>
            <dl>
              <div><dt>Latency</dt><dd>{activePoint.latency.toFixed(3)} <small>ms</small></dd></div>
              <div><dt>Throughput</dt><dd>{activePoint.fps.toFixed(1)} <small>FPS</small></dd></div>
              {activePoint.paramsM !== undefined && <div><dt>Parameters (FP32)</dt><dd>{activePoint.paramsM.toFixed(3)} <small>M</small></dd></div>}
              <div><dt>{datasetLabel} δ1</dt><dd>{activePoint.accuracy.toFixed(4)}</dd></div>
              {activePoint.absRel !== undefined && <div><dt>AbsRel</dt><dd>{activePoint.absRel.toFixed(4)}</dd></div>}
              {activePoint.rmse !== undefined && <div><dt>RMSE</dt><dd>{activePoint.rmse.toFixed(4)}</dd></div>}
            </dl>
          </> : <p>No matching benchmark rows.</p>}
          <div className="resolution-legend">
            {family === "relative" && <span><i className="shape-circle" />DepthART 224 × 224</span>}
            <span><i className="shape-diamond" />DepthART 448 × 448</span>
          </div>
        </aside>
      </div>

      {useStrict && family === "relative" && <div className="benchmark-comparison-status" aria-label="Additional comparison family status">
        <strong>Additional comparison families</strong>
        {additionalComparisonFamilies.map((item) => (
          <button
            type="button"
            key={item.label}
            className={`${item.className} ${activePoint?.methodFamily === item.label ? "active" : ""}`}
            disabled={!item.point}
            aria-pressed={activePoint?.methodFamily === item.label}
            onClick={() => item.point && setActive(item.point.id)}
            title={item.point ? `Highlight ${item.label} in the current plot` : `${item.label} is unavailable for this dataset`}
          >
            <i />{item.label}
          </button>
        ))}
        <em>{additionalComparisonFamilies.every((item) => item.point) ? "Included in the A6000 FP32 plot" : "Awaiting aligned A6000 FP32 model-only latency"}</em>
      </div>}

      <div className="benchmark-footnote">
        <div className="benchmark-encoder-legend">
          <strong>DepthART encoder</strong>
          {encoderOrder.map((encoder) => <span key={encoder}><i style={{ background: modelColors[encoder] }} />{encoderLabels[encoder]}</span>)}
        </div>
        <span className="fisheye-hint">Move across the plot to magnify nearby latency values.</span>
        {useStrict && family === "relative" && <span>Comparison methods use affine-invariant δ1.</span>}
      </div>
    </section>
  );
}
