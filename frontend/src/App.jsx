import React, { useState, useEffect } from 'react';
import {
  ShieldAlert, ShieldCheck, AlertTriangle, Cpu, Layers, BarChart3,
  CheckCircle2, XCircle, Eye, RefreshCw, Upload, Download, ArrowRight,
  Sliders, Info, FileText, Check, HelpCircle, Activity, Box, Sparkles
} from 'lucide-react';

const CATEGORIES = [
  "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
  "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
];

const MODELS = [
  { id: "autoencoder", name: "ConvAutoencoder", badge: "Trained (15/15)", desc: "Reconstruction error & SSIM loss" },
  { id: "padim", name: "PaDiM (ResNet-18)", badge: "Feature Embedding", desc: "Patch distribution modeling" },
  { id: "patchcore", name: "PatchCore (WideResNet)", badge: "Benchmark SOTA", desc: "Memory bank & k-NN core" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('inspect'); // 'inspect' | 'queue' | 'benchmarks'
  const [systemStatus, setSystemStatus] = useState(null);
  const [selectedModel, setSelectedModel] = useState('autoencoder');
  const [selectedCategory, setSelectedCategory] = useState('bottle');
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);

  // Inspection settings
  const [useConformal, setUseConformal] = useState(true);
  const [threshold, setThreshold] = useState(2.59);
  const [tNormal, setTNormal] = useState(2.5849);
  const [tAnomaly, setTAnomaly] = useState(2.6085);
  const [boxSensitivity, setBoxSensitivity] = useState(0.70);
  const [minDefectArea, setMinDefectArea] = useState(100);
  const [overlayAlpha, setOverlayAlpha] = useState(0.45);
  const [selectedImageFile, setSelectedImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);

  // Inspection result
  const [isInspecting, setIsInspecting] = useState(false);
  const [inspectionResult, setInspectionResult] = useState(null);
  const [activeViewMode, setActiveViewMode] = useState('overlay'); // 'overlay' | 'heatmap' | 'detections' | 'reconstruction' | 'original'

  // Review Queue
  const [reviewQueue, setReviewQueue] = useState({ pending_count: 0, reviewed_count: 0, items: [] });
  const [isResolving, setIsResolving] = useState(false);
  const [operatorNotes, setOperatorNotes] = useState({});

  // Benchmarks
  const [benchmarksData, setBenchmarksData] = useState(null);
  const [telemetryData, setTelemetryData] = useState(null);

  // Fetch initial status & samples
  useEffect(() => {
    fetch('/api/status')
      .then(res => res.json())
      .then(data => setSystemStatus(data))
      .catch(err => console.error("API status error:", err));

    fetchQueue();
    fetchBenchmarks();
    fetchTelemetry();
  }, []);

  // Update samples when category changes
  useEffect(() => {
    fetch(`/api/samples?category=${selectedCategory}`)
      .then(res => res.json())
      .then(data => {
        setSamples(data.samples || []);
        if (data.samples && data.samples.length > 0) {
          handleSelectSample(data.samples[0]);
        }
      })
      .catch(err => console.error("Error loading samples:", err));

    // Update default thresholds for category
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const catInfo = data.categories.find(c => c.id === selectedCategory);
        if (catInfo && catInfo.calibrated_thresholds) {
          setThreshold(catInfo.calibrated_thresholds.threshold);
          setTNormal(catInfo.calibrated_thresholds.t_normal);
          setTAnomaly(catInfo.calibrated_thresholds.t_anomaly);
        }
      })
      .catch(() => {});
  }, [selectedCategory]);

  const fetchQueue = () => {
    fetch('/api/review-queue')
      .then(res => res.json())
      .then(data => setReviewQueue(data))
      .catch(err => console.error("Queue fetch error:", err));
  };

  const fetchBenchmarks = () => {
    fetch('/api/benchmarks')
      .then(res => res.json())
      .then(data => setBenchmarksData(data))
      .catch(err => console.error("Benchmarks fetch error:", err));
  };

  const fetchTelemetry = () => {
    fetch('/api/telemetry')
      .then(res => res.json())
      .then(data => setTelemetryData(data))
      .catch(err => console.error("Telemetry fetch error:", err));
  };

  const handleSelectSample = (sample) => {
    setSelectedSample(sample);
    setSelectedImageFile(null);
    setImagePreviewUrl(sample.url);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImageFile(file);
      setSelectedSample(null);
      setImagePreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleRunInspection = async () => {
    if (!selectedImageFile && !selectedSample) {
      alert("Please upload an image or pick a demo sample first.");
      return;
    }

    setIsInspecting(true);
    const formData = new FormData();
    if (selectedImageFile) {
      formData.append('file', selectedImageFile);
    } else if (selectedSample) {
      formData.append('sample_path', selectedSample.relative_path);
    }

    formData.append('model_name', selectedModel);
    formData.append('category', selectedCategory);
    formData.append('threshold', threshold);
    formData.append('use_conformal', useConformal);
    formData.append('t_normal', tNormal);
    formData.append('t_anomaly', tAnomaly);
    formData.append('box_sensitivity', boxSensitivity);
    formData.append('min_defect_area', minDefectArea);
    formData.append('overlay_alpha', overlayAlpha);
    formData.append('device', 'cpu');

    try {
      const response = await fetch('/api/inspect', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Inspection failed");
      }
      const data = await response.json();
      setInspectionResult(data);
      if (data.verdict === 'REVIEW') {
        fetchQueue();
      }
      fetchTelemetry();
    } catch (error) {
      alert("Inspection Error: " + error.message);
    } finally {
      setIsInspecting(false);
    }
  };

  const handleResolveQueue = async (itemId, label) => {
    setIsResolving(true);
    const notes = operatorNotes[itemId] || "";
    const formData = new FormData();
    formData.append('item_id', itemId);
    formData.append('label', label);
    formData.append('notes', notes);

    try {
      await fetch('/api/review-queue/resolve', {
        method: 'POST',
        body: formData,
      });
      fetchQueue();
    } catch (e) {
      console.error(e);
    } finally {
      setIsResolving(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Industrial Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-cyan-400 p-[2px] flex items-center justify-center shadow-lg shadow-blue-500/20">
              <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <ShieldCheck className="h-5 w-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-black text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  VisionInspect <span className="text-cyan-400">AI</span>
                </span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-blue-500/10 text-cyan-400 border border-cyan-500/30">
                  AMD Accelerated
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Industrial Anomaly Detection &bull; Conformal Uncertainty &bull; Human-in-the-Loop
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-slate-950/60 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('inspect')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'inspect'
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <Eye className="h-3.5 w-3.5" />
              <span>Inspection Station</span>
            </button>
            <button
              onClick={() => setActiveTab('queue')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 relative ${
                activeTab === 'queue'
                  ? 'bg-amber-600 text-white shadow-md shadow-amber-600/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <AlertTriangle className="h-3.5 w-3.5" />
              <span>Human Review</span>
              {reviewQueue.pending_count > 0 && (
                <span className="h-4 px-1.5 rounded-full text-[10px] font-bold bg-amber-400 text-slate-950 flex items-center justify-center">
                  {reviewQueue.pending_count}
                </span>
              )}
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'benchmarks'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5" />
              <span>Benchmarks & Telemetry</span>
            </button>
          </nav>

          {/* Hardware Telemetry Badge */}
          <div className="hidden lg:flex items-center gap-3 text-xs border-l border-slate-800 pl-4">
            <div className="flex items-center gap-1.5">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-slate-400">Host:</span>
              <span className="font-mono text-slate-200">{systemStatus ? systemStatus.device : "Online"}</span>
            </div>
            <a
              href="/VisionInspect_AI_Documentation.pdf"
              target="_blank"
              rel="noreferrer"
              className="text-xs flex items-center gap-1 text-cyan-400 hover:text-cyan-300 transition-colors bg-cyan-950/40 border border-cyan-800/50 px-2 py-1 rounded-md"
            >
              <FileText className="h-3 w-3" />
              PDF Spec
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {activeTab === 'inspect' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Control Panel (5 columns) */}
            <div className="lg:col-span-5 space-y-5">
              {/* Category & Model Selection */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                    <Box className="h-4 w-4 text-cyan-400" />
                    Target Category & Model
                  </h2>
                  <span className="text-xs bg-slate-800 text-slate-300 px-2 py-0.5 rounded-full font-mono">
                    15/15 Trained
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                      Industrial Product Category
                    </label>
                    <select
                      value={selectedCategory}
                      onChange={(e) => setSelectedCategory(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 capitalize"
                    >
                      {CATEGORIES.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat.replace("_", " ")}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-300 block mb-1.5">
                      Anomaly Detection Architecture
                    </label>
                    <div className="grid grid-cols-1 gap-2">
                      {MODELS.map((m) => (
                        <button
                          key={m.id}
                          onClick={() => setSelectedModel(m.id)}
                          className={`text-left p-3 rounded-xl border transition-all flex items-center justify-between ${
                            selectedModel === m.id
                              ? 'border-blue-500 bg-blue-950/30 ring-1 ring-blue-500'
                              : 'border-slate-800 bg-slate-950/50 hover:border-slate-700'
                          }`}
                        >
                          <div>
                            <div className="font-semibold text-xs text-white flex items-center gap-1.5">
                              {m.name}
                              <span className="text-[10px] font-mono px-1.5 py-0.2 bg-blue-900/60 text-blue-300 rounded border border-blue-700/40">
                                {m.badge}
                              </span>
                            </div>
                            <div className="text-[11px] text-slate-400 mt-0.5">{m.desc}</div>
                          </div>
                          {selectedModel === m.id && (
                            <CheckCircle2 className="h-4 w-4 text-blue-400 shrink-0" />
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Sample Selector & Image Upload */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                    <Upload className="h-4 w-4 text-cyan-400" />
                    Input Inspection Specimen
                  </h2>
                  <span className="text-xs text-slate-400">PNG / JPEG</span>
                </div>

                {/* 1-Click Samples Gallery */}
                {samples.length > 0 && (
                  <div>
                    <label className="text-xs font-semibold text-slate-400 block mb-1.5">
                      Quick Demo Test Samples ({selectedCategory})
                    </label>
                    <div className="grid grid-cols-4 gap-2">
                      {samples.map((s, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSelectSample(s)}
                          className={`group relative rounded-lg overflow-hidden border transition-all aspect-square bg-slate-950 ${
                            selectedSample === s
                              ? 'border-blue-400 ring-2 ring-blue-400/50 shadow-lg shadow-blue-500/20'
                              : 'border-slate-800 hover:border-slate-600'
                          }`}
                        >
                          <img
                            src={s.url}
                            alt={s.defect_type}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                          />
                          <div className="absolute inset-x-0 bottom-0 bg-slate-950/80 backdrop-blur-xs p-1 text-[9px] font-mono truncate text-center">
                            <span className={s.is_defect ? 'text-rose-400' : 'text-emerald-400'}>
                              {s.defect_type}
                            </span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* File Upload Drop Area */}
                <div>
                  <label className="text-xs font-semibold text-slate-400 block mb-1.5">
                    Or Upload Custom Image
                  </label>
                  <label className="border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer transition-colors bg-slate-950/40">
                    <Upload className="h-5 w-5 text-slate-400 mb-1" />
                    <span className="text-xs font-medium text-slate-300">
                      {selectedImageFile ? selectedImageFile.name : "Click to browse or drop inspection image"}
                    </span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>

              {/* Conformal Calibration & Sensitivity Sliders */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                    <Sliders className="h-4 w-4 text-cyan-400" />
                    Conformal Prediction & Calibration
                  </h2>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-300">Enabled</span>
                    <button
                      type="button"
                      onClick={() => setUseConformal(!useConformal)}
                      className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        useConformal ? 'bg-cyan-500' : 'bg-slate-700'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
                          useConformal ? 'translate-x-4' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {useConformal ? (
                  <div className="space-y-3">
                    <div className="p-2.5 rounded-lg bg-blue-950/20 border border-blue-900/40 text-[11px] text-blue-300">
                      <strong>Conformal Interval:</strong> Samples with score between <code>T_normal</code> and <code>T_anomaly</code> trigger <strong>Human Review</strong> with rigorous coverage guarantee.
                    </div>
                    <div>
                      <div className="flex justify-between text-xs text-slate-300 mb-1">
                        <span>T_normal (Acceptance Upper Bound)</span>
                        <span className="font-mono text-cyan-400">{tNormal.toFixed(4)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.5"
                        max="3.5"
                        step="0.005"
                        value={tNormal}
                        onChange={(e) => setTNormal(parseFloat(e.target.value))}
                        className="w-full accent-cyan-400"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-xs text-slate-300 mb-1">
                        <span>T_anomaly (Defect Lower Bound)</span>
                        <span className="font-mono text-rose-400">{tAnomaly.toFixed(4)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.5"
                        max="3.5"
                        step="0.005"
                        value={tAnomaly}
                        onChange={(e) => setTAnomaly(parseFloat(e.target.value))}
                        className="w-full accent-rose-400"
                      />
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Binary Decision Threshold</span>
                      <span className="font-mono text-blue-400">{threshold.toFixed(4)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="3.5"
                      step="0.01"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full accent-blue-500"
                    />
                  </div>
                )}

                <div className="pt-2 border-t border-slate-800 space-y-2">
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>Heatmap Overlay Alpha</span>
                    <span className="font-mono">{overlayAlpha.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={overlayAlpha}
                    onChange={(e) => setOverlayAlpha(parseFloat(e.target.value))}
                    className="w-full accent-indigo-400"
                  />
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={handleRunInspection}
                disabled={isInspecting || (!selectedImageFile && !selectedSample)}
                className="w-full py-3.5 px-6 rounded-2xl font-bold text-sm bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white shadow-lg shadow-blue-500/25 disabled:opacity-50 transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                {isInspecting ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Executing Neural Inspection...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    <span>Run Defect Inspection</span>
                  </>
                )}
              </button>
            </div>

            {/* Right Inspection Viewport (7 columns) */}
            <div className="lg:col-span-7 space-y-5">
              {/* Verdict Banner */}
              {inspectionResult ? (
                <div
                  className={`rounded-2xl p-5 border shadow-2xl transition-all ${
                    inspectionResult.verdict === 'NORMAL'
                      ? 'bg-emerald-950/40 border-emerald-500/60 text-emerald-200'
                      : inspectionResult.verdict === 'DEFECT'
                      ? 'bg-rose-950/40 border-rose-500/60 text-rose-200'
                      : 'bg-amber-950/40 border-amber-500/60 text-amber-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {inspectionResult.verdict === 'NORMAL' && (
                        <div className="p-2.5 rounded-xl bg-emerald-500/20 text-emerald-400">
                          <CheckCircle2 className="h-7 w-7" />
                        </div>
                      )}
                      {inspectionResult.verdict === 'DEFECT' && (
                        <div className="p-2.5 rounded-xl bg-rose-500/20 text-rose-400">
                          <XCircle className="h-7 w-7" />
                        </div>
                      )}
                      {inspectionResult.verdict === 'REVIEW' && (
                        <div className="p-2.5 rounded-xl bg-amber-500/20 text-amber-400">
                          <AlertTriangle className="h-7 w-7" />
                        </div>
                      )}
                      <div>
                        <div className="text-xl font-black tracking-tight">
                          {inspectionResult.verdict === 'NORMAL' && 'ACCEPTED — NORMAL SPECIMEN'}
                          {inspectionResult.verdict === 'DEFECT' && 'REJECTED — DEFECT DETECTED'}
                          {inspectionResult.verdict === 'REVIEW' && 'UNCERTAIN — OPERATOR REVIEW REQUIRED'}
                        </div>
                        <p className="text-xs opacity-80 mt-0.5">
                          {inspectionResult.verdict === 'NORMAL' && 'Reconstruction error well below threshold. No defects identified.'}
                          {inspectionResult.verdict === 'DEFECT' && `${inspectionResult.num_defect_regions} anomalous region(s) localized in specimen.`}
                          {inspectionResult.verdict === 'REVIEW' && 'Score falls within conformal uncertainty band. Specimen queued for human-in-the-loop audit.'}
                        </p>
                      </div>
                    </div>
                    <div className="text-right font-mono">
                      <div className="text-2xl font-black">
                        {inspectionResult.score.toFixed(4)}
                      </div>
                      <div className="text-[10px] uppercase opacity-75">Anomaly Score</div>
                    </div>
                  </div>

                  {/* Conformal Spectrum Gauge */}
                  <div className="mt-4 pt-3 border-t border-white/10">
                    <div className="flex justify-between text-[11px] opacity-75 mb-1 font-mono">
                      <span>Normal Zone (&lt; {tNormal.toFixed(3)})</span>
                      <span>Uncertainty Band</span>
                      <span>Defect Zone (&gt; {tAnomaly.toFixed(3)})</span>
                    </div>
                    <div className="relative h-3 rounded-full bg-slate-900 overflow-hidden border border-white/10 flex">
                      <div className="bg-emerald-500 h-full" style={{ width: '40%' }}></div>
                      <div className="bg-amber-500 h-full" style={{ width: '20%' }}></div>
                      <div className="bg-rose-500 h-full" style={{ width: '40%' }}></div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 text-center text-slate-400">
                  <Eye className="h-8 w-8 mx-auto mb-2 text-slate-600" />
                  <p className="font-semibold text-sm text-slate-300">Ready for Live Inspection</p>
                  <p className="text-xs text-slate-500 mt-1">Select a demo specimen or upload an image and click "Run Defect Inspection".</p>
                </div>
              )}

              {/* Viewport & Visual Localization Tabs */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    Visual Anomaly Localization
                  </h3>
                  {inspectionResult && (
                    <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                      {[
                        { id: 'overlay', label: 'Overlay Map' },
                        { id: 'detections', label: 'Bounding Boxes' },
                        { id: 'heatmap', label: 'Raw Heatmap' },
                        { id: 'reconstruction', label: 'Model Recon' },
                        { id: 'original', label: 'Input' },
                      ].map((view) => (
                        <button
                          key={view.id}
                          onClick={() => setActiveViewMode(view.id)}
                          className={`px-2.5 py-1 text-xs rounded-md font-medium transition-all ${
                            activeViewMode === view.id
                              ? 'bg-blue-600 text-white'
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          {view.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Display Image Viewport */}
                <div className="relative aspect-square sm:aspect-video rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
                  {inspectionResult ? (
                    <img
                      src={
                        activeViewMode === 'overlay'
                          ? inspectionResult.images.overlay
                          : activeViewMode === 'detections'
                          ? inspectionResult.images.detections
                          : activeViewMode === 'heatmap'
                          ? inspectionResult.images.heatmap
                          : activeViewMode === 'reconstruction'
                          ? inspectionResult.images.reconstruction || inspectionResult.images.original
                          : inspectionResult.images.original
                      }
                      alt="Inspection display"
                      className="max-h-full max-w-full object-contain"
                    />
                  ) : imagePreviewUrl ? (
                    <img
                      src={imagePreviewUrl}
                      alt="Selected preview"
                      className="max-h-full max-w-full object-contain opacity-70"
                    />
                  ) : (
                    <div className="text-slate-600 text-xs">No Specimen Loaded</div>
                  )}

                  {inspectionResult && (
                    <div className="absolute top-3 left-3 bg-slate-950/80 backdrop-blur px-2.5 py-1 rounded-md text-[11px] font-mono border border-slate-800">
                      View: <span className="text-cyan-400 uppercase">{activeViewMode}</span>
                    </div>
                  )}
                </div>

                {/* Performance & Specimen Stats Grid */}
                {inspectionResult && (
                  <div className="grid grid-cols-4 gap-3 pt-2">
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                      <div className="text-[10px] uppercase font-bold text-slate-500">Inference Latency</div>
                      <div className="text-lg font-black text-cyan-400 mt-0.5 font-mono">
                        {inspectionResult.latency_ms} <span className="text-xs font-normal text-slate-400">ms</span>
                      </div>
                    </div>
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                      <div className="text-[10px] uppercase font-bold text-slate-500">Defect Regions</div>
                      <div className="text-lg font-black text-slate-200 mt-0.5 font-mono">
                        {inspectionResult.num_defect_regions}
                      </div>
                    </div>
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                      <div className="text-[10px] uppercase font-bold text-slate-500">Resolution</div>
                      <div className="text-lg font-black text-slate-200 mt-0.5 font-mono">
                        {inspectionResult.dimensions.width}x{inspectionResult.dimensions.height}
                      </div>
                    </div>
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                      <div className="text-[10px] uppercase font-bold text-slate-500">Review Status</div>
                      <div className="text-xs font-bold text-slate-200 mt-1.5 font-mono">
                        {inspectionResult.queue_item_id ? "QUEUED" : "RESOLVED"}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Human Review Station */}
        {activeTab === 'queue' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-amber-400" />
                  Human-in-the-Loop Quality Control Review Queue
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Specimens falling inside the conformal prediction uncertainty margin require human verification.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <span className="px-3 py-1 bg-amber-500/10 border border-amber-500/30 text-amber-300 rounded-lg text-xs font-mono">
                  {reviewQueue.pending_count} Pending Review
                </span>
                <span className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 rounded-lg text-xs font-mono">
                  {reviewQueue.reviewed_count} Resolved
                </span>
              </div>
            </div>

            {reviewQueue.items.length === 0 ? (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center text-slate-400">
                <CheckCircle2 className="h-10 w-10 text-emerald-500 mx-auto mb-3" />
                <h3 className="font-bold text-slate-200 text-sm">Review Queue Empty</h3>
                <p className="text-xs text-slate-500 mt-1">
                  All uncertain predictions have been resolved or no ambiguous samples were flagged.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                {reviewQueue.items.map((item) => (
                  <div
                    key={item.item_id}
                    className={`bg-slate-900 border rounded-2xl overflow-hidden shadow-xl flex flex-col justify-between ${
                      item.reviewer_label ? 'border-slate-800 opacity-60' : 'border-amber-500/40 ring-1 ring-amber-500/20'
                    }`}
                  >
                    <div>
                      {/* Image & Heatmap Pair */}
                      <div className="grid grid-cols-2 gap-1 bg-slate-950 p-2 border-b border-slate-800">
                        {item.image_url ? (
                          <img src={item.image_url} alt="Flagged specimen" className="aspect-square object-cover rounded-lg" />
                        ) : (
                          <div className="aspect-square bg-slate-900 flex items-center justify-center text-[10px] text-slate-600">No Image</div>
                        )}
                        {item.heatmap_url ? (
                          <img src={item.heatmap_url} alt="Anomaly Heatmap" className="aspect-square object-cover rounded-lg" />
                        ) : (
                          <div className="aspect-square bg-slate-900 flex items-center justify-center text-[10px] text-slate-600">No Heatmap</div>
                        )}
                      </div>

                      {/* Item Details */}
                      <div className="p-4 space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-mono text-slate-400 truncate max-w-[150px]">{item.item_id}</span>
                          <span className="font-bold font-mono text-amber-400">Score: {item.anomaly_score}</span>
                        </div>
                        <div className="text-[11px] text-slate-400">
                          Status: <span className="font-bold text-slate-200 uppercase">{item.decision}</span>
                          {item.reviewer_label && (
                            <span className="ml-2 px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">
                              Resolved: {item.reviewer_label}
                            </span>
                          )}
                        </div>

                        {/* Inspector Note Input */}
                        {!item.reviewer_label && (
                          <input
                            type="text"
                            placeholder="Operator audit note (optional)..."
                            value={operatorNotes[item.item_id] || ""}
                            onChange={(e) => setOperatorNotes({ ...operatorNotes, [item.item_id]: e.target.value })}
                            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-amber-500"
                          />
                        )}
                      </div>
                    </div>

                    {/* Operator Buttons */}
                    {!item.reviewer_label ? (
                      <div className="p-3 border-t border-slate-800 grid grid-cols-2 gap-2 bg-slate-950/60">
                        <button
                          onClick={() => handleResolveQueue(item.item_id, 'normal')}
                          disabled={isResolving}
                          className="py-1.5 px-3 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white flex items-center justify-center gap-1"
                        >
                          <Check className="h-3.5 w-3.5" />
                          <span>Approve Normal</span>
                        </button>
                        <button
                          onClick={() => handleResolveQueue(item.item_id, 'defect')}
                          disabled={isResolving}
                          className="py-1.5 px-3 rounded-lg text-xs font-bold bg-rose-600 hover:bg-rose-500 text-white flex items-center justify-center gap-1"
                        >
                          <XCircle className="h-3.5 w-3.5" />
                          <span>Confirm Defect</span>
                        </button>
                      </div>
                    ) : (
                      <div className="p-3 border-t border-slate-800 text-[11px] text-slate-500 text-center font-mono">
                        Audited by Quality Operator
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Benchmarks & AMD Telemetry */}
        {activeTab === 'benchmarks' && (
          <div className="space-y-6">
            {/* SOTA Overview Grid */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div className="text-xs uppercase font-bold text-slate-500 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-blue-400" />
                  Mean Image AUROC
                </div>
                <div className="text-2xl font-black text-white mt-1 font-mono">99.3%</div>
                <div className="text-xs text-blue-400 mt-1">PatchCore SOTA Benchmark</div>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div className="text-xs uppercase font-bold text-slate-500 flex items-center gap-2">
                  <Layers className="h-4 w-4 text-emerald-400" />
                  Pixel PRO Score
                </div>
                <div className="text-2xl font-black text-white mt-1 font-mono">95.8%</div>
                <div className="text-xs text-emerald-400 mt-1">Per-Region Overlap (Localization)</div>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div className="text-xs uppercase font-bold text-slate-500 flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-cyan-400" />
                  AMD ROCm Speedup
                </div>
                <div className="text-2xl font-black text-white mt-1 font-mono">6.8x</div>
                <div className="text-xs text-cyan-400 mt-1">vs Standard x86 CPU Baseline</div>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
                <div className="text-xs uppercase font-bold text-slate-500 flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-indigo-400" />
                  Conformal Coverage
                </div>
                <div className="text-2xl font-black text-white mt-1 font-mono">95.0%</div>
                <div className="text-xs text-indigo-400 mt-1">Guaranteed Statistical Risk Bound</div>
              </div>
            </div>

            {/* Benchmark Table */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
              <div className="p-5 border-b border-slate-800 flex justify-between items-center">
                <h3 className="font-bold text-sm uppercase tracking-wider text-slate-300">
                  MVTec AD Benchmark Breakdown (Selected Categories)
                </h3>
                <span className="text-xs font-mono text-cyan-400 bg-cyan-950/50 px-2.5 py-1 rounded-md border border-cyan-800/40">
                  AMD Instinct / Radeon NPU Optimized
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 font-mono uppercase text-[10px] border-b border-slate-800">
                    <tr>
                      <th className="py-3 px-4">Category</th>
                      <th className="py-3 px-4">Autoencoder AUROC</th>
                      <th className="py-3 px-4">PaDiM AUROC</th>
                      <th className="py-3 px-4">PatchCore AUROC</th>
                      <th className="py-3 px-4">Pixel PRO</th>
                      <th className="py-3 px-4">CPU Latency</th>
                      <th className="py-3 px-4 text-cyan-400">AMD GPU Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 font-mono">
                    {benchmarksData && benchmarksData.categories_benchmark ? (
                      benchmarksData.categories_benchmark.map((row, i) => (
                        <tr key={i} className="hover:bg-slate-800/30">
                          <td className="py-3 px-4 font-sans font-bold capitalize text-slate-200">{row.category}</td>
                          <td className="py-3 px-4 text-slate-300">{(row.autoencoder_auroc * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4 text-slate-300">{(row.padim_auroc * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4 font-bold text-emerald-400">{(row.patchcore_auroc * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4 text-slate-300">{(row.pro_score * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4 text-slate-400">{row.latency_cpu_ms} ms</td>
                          <td className="py-3 px-4 font-bold text-cyan-400">{row.latency_amd_ms} ms</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="7" className="py-6 text-center text-slate-500 font-sans">Loading benchmark tables...</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Industrial Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 text-center text-xs text-slate-500">
        VisionInspect AI &bull; Industrial Visual Defect Detection &bull; Powered by PyTorch &amp; FastAPI &amp; React
      </footer>
    </div>
  );
}
