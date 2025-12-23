import React, { useState, useCallback } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { 
  Upload, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Cpu, 
  Database,
  ChevronRight,
  RefreshCw,
  AlertCircle
} from 'lucide-react';
import { PredictionModel, GoldDataPoint, PredictionResult, FileData } from './types';
//import { analyzeGoldPrices } from './services/geminiService';
import { predictLSTMFromBackend } from './services/predictService';
import { trainLSTMFromBackend } from './services/trainService';
import { runGM11, runGM11TestPredict } from './utils/greyModel';

type TrainOutcome = {
  loss?: number;
  mape?: number;
  metadata?: Record<string, unknown>;
  raw: Record<string, unknown>;
};

const App: React.FC = () => {
  const [fileData, setFileData] = useState<FileData | null>(null);
  const [selectedModel, setSelectedModel] = useState<PredictionModel>(PredictionModel.LSTM);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  //Thêm useState LSTML
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [retrainFile, setRetrainFile] = useState<File | null>(null);
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [retrainError, setRetrainError] = useState<string | null>(null);
  const [retrainResult, setRetrainResult] = useState<TrainOutcome | null>(null);
  const [detectedColumns, setDetectedColumns] = useState<{dateCol: string, priceCol: string} | null>(null);
  const [retrainDetectedColumns, setRetrainDetectedColumns] = useState<{dateCol: string, priceCol: string} | null>(null);
  
  // Helper to clean column names (remove quotes and trim)
  const cleanColumnName = (col: string): string => {
    return col.replace(/^["']|["']$/g, '').trim();
  };

  // Helper to detect date and price columns from headers
  const detectColumns = (headers: string[]): {dateCol: string, priceCol: string} => {
    const cleanedHeaders = headers.map(cleanColumnName);
    
    // Detect date column: first column containing "date" (case-insensitive) or first column
    let dateCol = cleanedHeaders.find(h => h.toLowerCase().includes('date')) || cleanedHeaders[0];
    
    // Detect price column: column containing "price" or "close" (case-insensitive) or second column
    let priceCol = cleanedHeaders.find(h => 
      h.toLowerCase().includes('price') || h.toLowerCase().includes('close')
    ) || cleanedHeaders[1] || cleanedHeaders[0];
    
    return { dateCol, priceCol };
  };

  // Helper to parse CSV lines that contain quotes and commas
  const parseCSVLine = (line: string) => {
    const result = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    result.push(current.trim());
    return result;
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadedFile(file); // ✅ THÊM DÒNG NÀY LSTM

    const reader = new FileReader();
    reader.onload = (e) => {
      let content = e.target?.result as string;
      
      // Remove UTF-8 BOM if present
      if (content.charCodeAt(0) === 0xFEFF) {
        content = content.slice(1);
      }

      try {
        const lines = content.split(/\r?\n/);
        if (lines.length < 2) throw new Error("File empty or invalid");

        // Parse and detect columns from header
        const headerLine = lines[0];
        const headers = parseCSVLine(headerLine);
        const detected = detectColumns(headers);
        setDetectedColumns(detected);

        const parsed: GoldDataPoint[] = lines
          .slice(1) // Skip header
          .filter(line => line.trim() !== '')
          .map(line => {
            const columns = parseCSVLine(line);
            const dateStr = columns[0];
            // Remove commas from price string like "1,826.20"
            const priceStr = columns[1]?.replace(/,/g, '');
            
            return {
              date: dateStr,
              price: parseFloat(priceStr)
            };
          })
          .filter(p => !isNaN(p.price) && p.date);

        if (parsed.length === 0) throw new Error("No valid data found");

        // Sort data by date ascending (CSV is often descending)
        const sortedData = parsed.sort((a, b) => 
          new Date(a.date).getTime() - new Date(b.date).getTime()
        );
        
        setFileData({
          name: file.name,
          content,
          parsed: sortedData
        });
        setResult(null); // Reset result when new file is uploaded
        setError(null);
      } catch (err) {
        console.error(err);
        setError("Lỗi định dạng file. Vui lòng sử dụng CSV có cột 'Date' và 'Price'.");
      }
    };
    reader.readAsText(file);
  };

  const handleRetrainFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setRetrainFile(file);
    setRetrainResult(null);
    setRetrainError(null);

    // Detect columns from retrain file
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      try {
        const lines = content.split(/\r?\n/);
        if (lines.length > 0) {
          const headerLine = lines[0].charCodeAt(0) === 0xFEFF ? lines[0].slice(1) : lines[0];
          const headers = parseCSVLine(headerLine);
          const detected = detectColumns(headers);
          setRetrainDetectedColumns(detected);
        }
      } catch (err) {
        console.error("Failed to detect columns:", err);
      }
    };
    reader.readAsText(file);
  };

  const handleRetrain = async () => {
    if (!retrainFile) return;
    setRetrainLoading(true);
    setRetrainError(null);
    setRetrainResult(null);

    try {
      const cols = retrainDetectedColumns || { dateCol: "Date", priceCol: "Price" };
      const response = await trainLSTMFromBackend({
        file: retrainFile,
        dateCol: cols.dateCol,
        priceCol: cols.priceCol,
        windowSize: 60,
      });

      const responseObj: Record<string, unknown> =
        response && typeof response === "object" ? response : { result: response };
      const lossValue = responseObj["loss"];
      const mapeValue = responseObj["mape"] ?? responseObj["MAPE"];
      const metadataValue = responseObj["metadata"];

      setRetrainResult({
        loss: typeof lossValue === "number" ? lossValue : undefined,
        mape: typeof mapeValue === "number" ? mapeValue : undefined,
        metadata:
          metadataValue && typeof metadataValue === "object"
            ? (metadataValue as Record<string, unknown>)
            : undefined,
        raw: responseObj,
      });
    } catch (err: any) {
      setRetrainError(err?.message || "Đã xảy ra lỗi khi retrain.");
    } finally {
      setRetrainLoading(false);
    }
  };

const handlePredict = async () => {
  if (!fileData) return;
  setLoading(true);
  setError(null);

  try {
    const getLastYear = (data: GoldDataPoint[]) => {
      const years = data.map(d => new Date(d.date).getFullYear());
      return Math.max(...years);
    };

    const splitByLastYear = (data: GoldDataPoint[]) => {
      const lastYear = getLastYear(data);
      const train = data.filter(d => new Date(d.date).getFullYear() < lastYear);
      const test  = data.filter(d => new Date(d.date).getFullYear() === lastYear);
      return { lastYear, train, test };
    };

    // GM11 - local
    if (selectedModel === PredictionModel.GM11) {
      const { lastYear, train, test } = splitByLastYear(fileData.parsed);

      if (train.length < 4) {
        throw new Error("Dữ liệu train quá ít cho GM(1,1)");
      }
      if (test.length === 0) {
        throw new Error("Không có dữ liệu test (năm cuối)");
      }

      // Predict for test dates using model fitted on train only
      const forecast = runGM11TestPredict(train, test);

      // Map test actual data
      const testActual = test.map(p => ({
        date: p.date,
        price: p.price,
        isPrediction: false,
      }));

      // Calculate trend
      const lastActual = test[test.length - 1].price;
      const lastPred = forecast[forecast.length - 1].price;
      const trend = lastPred > lastActual ? "up" : lastPred < lastActual ? "down" : "flat";

      // Simple accuracy estimation
      const actualPrices = test.map(p => p.price);
      const predPrices = forecast.map(p => p.price);
      const errors = actualPrices.map((a, i) => Math.abs(a - predPrices[i]) / a);
      const avgError = errors.reduce((sum, e) => sum + e, 0) / errors.length;
      const pseudoAcc = Math.max(0, 1 - avgError);

      setResult({
        historical: train,
        forecast,
        testActual,
        metrics: {
          trend,
          accuracy: pseudoAcc,
          confidence: 0.7,
        } as any,
      } as any);

      setLoading(false);
      return;
    }

    // LSTM - backend
    if (!uploadedFile) throw new Error("Chưa có file upload để gửi backend.");

    const { lastYear, train } = splitByLastYear(fileData.parsed);

    const cols = detectedColumns || { dateCol: "Date", priceCol: "Price" };
    const backend = await predictLSTMFromBackend({
      file: uploadedFile,
      dateCol: cols.dateCol,
      priceCol: cols.priceCol,
      windowSize: 60,
      testYear: lastYear,
    });

    // Use only train data as historical (not full dataset)
    const historical = train;

    // Map test actual data using backend dates
    const testActual = backend.dates.map((d: string, i: number) => ({
      date: d,
      price: backend.actual[i],
      isPrediction: false,
    })) as any;

    // Map predicted data
    const forecast = backend.dates.map((d, i) => ({
      date: d,
      price: backend.predicted[i],
      isPrediction: true,
    })) as any;

    // Calculate trend from test data
    const lastActual = backend.actual[backend.actual.length - 1];
    const lastPred = backend.predicted[backend.predicted.length - 1];
    const trend = lastPred > lastActual ? "up" : lastPred < lastActual ? "down" : "flat";

    const pseudoAcc = Math.max(0, 1 - backend.mape / 100);
    const pseudoConf = Math.max(0.2, Math.min(0.95, pseudoAcc));

    setResult({
      historical,
      forecast,
      testActual,
      metrics: {
        trend,
        accuracy: pseudoAcc,
        confidence: pseudoConf,
      } as any,
    } as any);
  } catch (err: any) {
    setError(err?.message || "Đã xảy ra lỗi trong quá trình dự đoán.");
  } finally {
    setLoading(false);
  }
};

  const predByDate = new Map(
    (result?.forecast ?? []).map((x: any) => [x.date, x.price])
  );

  const chartData = result
    ? [
        // Last 60 train points
        ...result.historical.slice(-60).map(p => ({
          date: p.date,
          trainActual: p.price,
          testActual: null,
          testPredicted: null,
        })),
        // Test actual & predicted (align by date)
        ...(result.testActual ?? []).map((p: any) => ({
          date: p.date,
          trainActual: null,
          testActual: p.price,
          testPredicted: predByDate.get(p.date) ?? null,
        })),
      ]
    : fileData?.parsed.slice(-60).map(p => ({
        date: p.date,
        trainActual: p.price,
        testActual: null,
        testPredicted: null,
      })) || []; // Ensure this line is properly closed

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-amber-500 p-2 rounded-lg">
              <Activity className="text-white w-6 h-6" />
            </div>
            <h1 className="text-xl font-bold text-slate-800">GoldPrice<span className="text-amber-600">AI</span></h1>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <span className="text-sm font-medium text-slate-400">Dự đoán giá vàng thông minh</span>
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full p-4 md:p-8 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* File Upload */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Database className="w-5 h-5 text-amber-600" />
                <h2 className="font-semibold text-slate-800">Dữ liệu đầu vào</h2>
              </div>
              <p className="text-sm text-slate-500 mb-6">Tải lên file CSV lịch sử giá vàng (Hỗ trợ định dạng Investing.com).</p>
            </div>
            
            <label className="group flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-200 rounded-xl cursor-pointer hover:bg-slate-50 transition-all">
              <div className="flex flex-col items-center justify-center pt-5 pb-6 px-4 text-center">
                <Upload className="w-8 h-8 text-slate-400 group-hover:text-amber-500 mb-2" />
                <p className="text-xs text-slate-500 group-hover:text-slate-700 truncate w-full">
                  {fileData ? fileData.name : 'Nhấn để chọn file CSV'}
                </p>
              </div>
              <input type="file" className="hidden" accept=".csv" onChange={handleFileUpload} />
            </label>
            {detectedColumns && (
              <div className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
                <p className="text-xs font-semibold text-slate-600 mb-1">Cột phát hiện:</p>
                <p className="text-xs text-slate-500">📅 Date: <span className="font-medium text-slate-700">{detectedColumns.dateCol}</span></p>
                <p className="text-xs text-slate-500">💰 Price: <span className="font-medium text-slate-700">{detectedColumns.priceCol}</span></p>
              </div>
            )}
          </div>

          {/* Model Choice */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Cpu className="w-5 h-5 text-amber-600" />
                <h2 className="font-semibold text-slate-800">Mô hình dự đoán</h2>
              </div>
              <div className="space-y-3">
                <button
                  onClick={() => setSelectedModel(PredictionModel.LSTM)}
                  className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all ${
                    selectedModel === PredictionModel.LSTM 
                    ? 'border-amber-500 bg-amber-50 text-amber-900 ring-2 ring-amber-200' 
                    : 'border-slate-100 bg-slate-50 text-slate-600 hover:border-slate-300'
                  }`}
                >
                  <span className="font-medium">LSTM (Deep Learning)</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setSelectedModel(PredictionModel.GM11)}
                  className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all ${
                    selectedModel === PredictionModel.GM11 
                    ? 'border-amber-500 bg-amber-50 text-amber-900 ring-2 ring-amber-200' 
                    : 'border-slate-100 bg-slate-50 text-slate-600 hover:border-slate-300'
                  }`}
                >
                  <span className="font-medium">GM (1,1) Grey Model</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Action */}
          <div className="bg-slate-900 p-6 rounded-2xl shadow-xl flex flex-col justify-between relative overflow-hidden">
            <div className="absolute top-0 right-0 p-8 opacity-10">
              <Activity className="w-32 h-32 text-amber-500" />
            </div>
            <div className="relative z-10">
              <h2 className="text-white font-semibold text-lg mb-2">Thực hiện dự báo</h2>
              <p className="text-slate-400 text-sm mb-6">Hệ thống sẽ phân tích xu hướng dựa trên mô hình {selectedModel}.</p>
            </div>
            <button
              disabled={!fileData || loading}
              onClick={handlePredict}
              className={`relative z-10 w-full flex items-center justify-center gap-2 py-3 px-6 rounded-xl font-bold transition-all ${
                !fileData || loading 
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                : 'bg-amber-500 hover:bg-amber-400 text-white shadow-lg shadow-amber-500/20 active:scale-[0.98]'
              }`}
            >
              {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : 'Bắt đầu dự đoán'}
            </button>
          </div>
        </div>

        <div className="bg-white p-6 md:p-8 rounded-3xl shadow-sm border border-slate-100 space-y-6">
          <div className="flex items-center gap-2">
            <RefreshCw className="w-5 h-5 text-amber-600" />
            <h2 className="font-semibold text-slate-800">Retrain mô hình LSTM</h2>
          </div>
          <p className="text-sm text-slate-500">
            Tải lên dữ liệu mới để huấn luyện lại mô hình và nhận thông tin loss/mape/metadata.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-stretch">
            <div className="flex flex-col gap-2">
              <label className="group flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-200 rounded-xl cursor-pointer hover:bg-slate-50 transition-all">
                <div className="flex flex-col items-center justify-center pt-5 pb-6 px-4 text-center">
                  <Upload className="w-8 h-8 text-slate-400 group-hover:text-amber-500 mb-2" />
                  <p className="text-xs text-slate-500 group-hover:text-slate-700 truncate w-full">
                    {retrainFile ? retrainFile.name : 'Chọn file CSV cho retrain'}
                  </p>
                </div>
                <input type="file" className="hidden" accept=".csv" onChange={handleRetrainFileUpload} />
              </label>
              {retrainDetectedColumns && (
                <div className="p-2 bg-slate-50 rounded-lg border border-slate-200">
                  <p className="text-xs font-semibold text-slate-600 mb-1">Cột phát hiện:</p>
                  <p className="text-xs text-slate-500">📅 {retrainDetectedColumns.dateCol}</p>
                  <p className="text-xs text-slate-500">💰 {retrainDetectedColumns.priceCol}</p>
                </div>
              )}
            </div>

            <button
              disabled={!retrainFile || retrainLoading}
              onClick={handleRetrain}
              className={`w-full flex items-center justify-center gap-2 py-3 px-6 rounded-xl font-bold transition-all ${
                !retrainFile || retrainLoading
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  : 'bg-amber-500 hover:bg-amber-400 text-white shadow-lg shadow-amber-500/20 active:scale-[0.98]'
              }`}
            >
              {retrainLoading ? <RefreshCw className="w-5 h-5 animate-spin" /> : 'Retrain'}
            </button>

            <div className="flex flex-col justify-center rounded-xl border border-slate-100 bg-slate-50 p-4">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Trạng thái</span>
              <span className={`mt-2 text-sm font-semibold ${
                retrainLoading ? 'text-amber-600' : retrainError ? 'text-rose-600' : retrainResult ? 'text-emerald-600' : 'text-slate-500'
              }`}>
                {retrainLoading
                  ? 'Đang huấn luyện...'
                  : retrainError
                    ? 'Thất bại'
                    : retrainResult
                      ? 'Thành công'
                      : 'Chưa chạy'}
              </span>
              {retrainError && <p className="mt-2 text-xs text-rose-500">{retrainError}</p>}
            </div>
          </div>

          {retrainResult && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="rounded-2xl border border-slate-100 p-4">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Loss</span>
                <div className="mt-2 text-lg font-bold text-slate-800">
                  {typeof retrainResult.loss === 'number' ? retrainResult.loss.toFixed(6) : 'N/A'}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-100 p-4">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">MAPE</span>
                <div className="mt-2 text-lg font-bold text-slate-800">
                  {typeof retrainResult.mape === 'number' ? `${retrainResult.mape.toFixed(4)}%` : 'N/A'}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-100 p-4">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Metadata</span>
                <pre className="mt-2 text-xs text-slate-600 whitespace-pre-wrap break-words">
                  {JSON.stringify(retrainResult.metadata ?? retrainResult.raw, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p className="text-sm font-medium">{error}</p>
          </div>
        )}

        {loading && (
          <div className="p-20 bg-white rounded-3xl border border-slate-100 shadow-sm flex flex-col items-center gap-4 text-center">
             <div className="w-16 h-16 border-4 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
             <p className="text-slate-500 font-medium animate-pulse">Đang xử lý dữ liệu bằngAI...</p>
          </div>
        )}

        {result && !loading && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
            {/* Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Xu hướng</span>
                <div className="flex items-center gap-2 mt-1">
                  {result.metrics.trend === 'up' ? (
                    <>
                      <TrendingUp className="text-emerald-500 w-5 h-5" />
                      <span className="text-lg font-bold text-emerald-600">Tăng (Bullish)</span>
                    </>
                  ) : result.metrics.trend === 'down' ? (
                    <>
                      <TrendingDown className="text-rose-500 w-5 h-5" />
                      <span className="text-lg font-bold text-rose-600">Giảm (Bearish)</span>
                    </>
                  ) : (
                    <span className="text-lg font-bold text-slate-600">Đi ngang</span>
                  )}
                </div>
              </div>
              <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Độ chính xác (R²)</span>
                <div className="mt-1">
                  <span className="text-2xl font-bold text-slate-800">{(result.metrics.accuracy * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Giá hiện tại</span>
                <div className="mt-1">
                  <span className="text-2xl font-bold text-slate-800">${result.historical[result.historical.length-1].price.toLocaleString()}</span>
                </div>
              </div>
              <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Độ tin cậy AI</span>
                <div className="w-full bg-slate-100 h-2 rounded-full mt-3 overflow-hidden">
                  <div 
                    className="h-full bg-amber-500 rounded-full transition-all duration-1000" 
                    style={{ width: `${result.metrics.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Chart */}
            <div className="bg-white p-6 md:p-8 rounded-3xl shadow-sm border border-slate-100">
              <div className="flex justify-between items-center mb-8">
                <div>
                  <h3 className="text-lg font-bold text-slate-800">Biểu đồ dự báo</h3>
                  <p className="text-sm text-slate-500">60 phiên train gần nhất và dữ liệu test năm {result?.testActual?.[0] ? new Date(result.testActual[0].date).getFullYear() : 'cuối'}</p>
                </div>
              </div>

              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="colorTrain" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#64748b" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#64748b" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="colorTestActual" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis 
                      dataKey="date" 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: '#94a3b8', fontSize: 10 }} 
                      dy={10}
                    />
                    <YAxis 
                      domain={['auto', 'auto']} 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: '#94a3b8', fontSize: 12 }} 
                    />
                    <Tooltip 
                      contentStyle={{ 
                        borderRadius: '12px', 
                        border: 'none', 
                        boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)',
                        padding: '12px'
                      }}
                    />
                    <Legend verticalAlign="top" height={36} />
                    <Area
                      type="monotone"
                      dataKey="trainActual"
                      stroke="#64748b"
                      strokeWidth={2}
                      fill="url(#colorTrain)"
                      name="Train (Lịch sử)"
                      connectNulls
                    />
                    <Area
                      type="monotone"
                      dataKey="testActual"
                      stroke="#3b82f6"
                      strokeWidth={3}
                      fill="url(#colorTestActual)"
                      name="Test Actual"
                      connectNulls
                    />
                    <Area
                      type="monotone"
                      dataKey="testPredicted"
                      stroke="#f59e0b"
                      strokeWidth={3}
                      fillOpacity={0}
                      connectNulls
                      strokeDasharray="6 6"
                      name="Test Predicted"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Table */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
              <div className="p-6 border-b border-slate-100">
                <h3 className="font-bold text-slate-800">Chi tiết giá dự báo vs Thực tế</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead className="bg-slate-50 text-[10px] uppercase font-bold text-slate-400">
                    <tr>
                      <th className="px-6 py-3">Ngày</th>
                      <th className="px-6 py-3">Giá Thực Tế (Test)</th>
                      <th className="px-6 py-3">Giá Dự Báo</th>
                      <th className="px-6 py-3">Sai Số</th>
                      <th className="px-6 py-3">Sai Số %</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {(result.testActual ?? []).map((actual: any, idx: number) => {
                      const predicted = result.forecast[idx];
                      const error = predicted ? Math.abs(actual.price - predicted.price) : 0;
                      const errorPercent = predicted ? (error / actual.price) * 100 : 0;
                      
                      return (
                        <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                          <td className="px-6 py-4 text-sm font-medium text-slate-700">{actual.date}</td>
                          <td className="px-6 py-4 text-sm font-bold text-blue-600">
                            ${actual.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                          </td>
                          <td className="px-6 py-4 text-sm font-bold text-amber-600">
                            ${predicted ? predicted.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : 'N/A'}
                          </td>
                          <td className="px-6 py-4 text-sm font-medium text-slate-700">
                            ${error.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                          </td>
                          <td className="px-6 py-4 text-sm font-medium">
                            <span className={`inline-flex items-center px-2 py-1 rounded-md font-bold text-[10px] ${
                              errorPercent < 5 
                                ? 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200' 
                                : errorPercent < 10
                                ? 'bg-yellow-50 text-yellow-700 ring-1 ring-yellow-200'
                                : 'bg-red-50 text-red-700 ring-1 ring-red-200'
                            }`}>
                              {errorPercent.toFixed(2)}%
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {!fileData && !loading && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mb-6">
              <Database className="w-10 h-10 text-slate-300" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-2">Chưa có dữ liệu</h3>
            <p className="text-slate-500 max-w-sm mx-auto">
              Vui lòng tải lên file CSV để bắt đầu quá trình phân tích và dự báo giá vàng.
            </p>
          </div>
        )}
      </main>

      <footer className="bg-white border-t border-slate-200 py-8 px-6 text-center text-sm text-slate-400">
        <p>&copy; 2025 GoldPriceAI Predictor. Powered by Đỗ Phúc Vũ Hà.</p>
      </footer>
    </div>
  );
};

export default App;
