import { GoldDataPoint } from '../types';

/**
 * Basic GM(1,1) Implementation for Time Series Prediction
 */
export const runGM11 = (data: GoldDataPoint[], numForecast: number): GoldDataPoint[] => {
  const prices = data.map(d => d.price);
  const N = prices.length;
  if (N < 3 || !isFinite(prices[0])) return [];

  // AGO (x1)
  const x1: number[] = [prices[0]];
  for (let i = 1; i < N; i++) x1.push(x1[i - 1] + prices[i]);

  // Background values z (length N-1) and Y (x0 from 2..N)
  const z: number[] = [];
  for (let i = 0; i < N - 1; i++) z.push(0.5 * (x1[i] + x1[i + 1]));
  const y: number[] = prices.slice(1);

  // Least squares for y = c*z + b, GM uses y = -a*z + b => a = -c
  let sumZ = 0, sumY = 0, sumZZ = 0, sumZY = 0;
  for (let i = 0; i < N - 1; i++) {
    const zi = z[i], yi = y[i];
    sumZ += zi; sumY += yi; sumZZ += zi * zi; sumZY += zi * yi;
  }
  const n = N - 1;
  const denom = n * sumZZ - sumZ * sumZ;
  const EPS = 1e-12;
  if (Math.abs(denom) < EPS) return [];

  const c = (n * sumZY - sumZ * sumY) / denom;
  const b = (sumY - c * sumZ) / n;
  const a = -c;

  // x1_hat for k = 1..(N + numForecast)
  const total = N + Math.max(0, numForecast);
  const x1_hat: number[] = new Array(total);
  if (Math.abs(a) < EPS) {
    for (let k = 1; k <= total; k++) x1_hat[k - 1] = prices[0] + b * (k - 1);
  } else {
    const C = prices[0] - b / a;
    for (let k = 1; k <= total; k++) x1_hat[k - 1] = C * Math.exp(-a * (k - 1)) + b / a;
  }

  // Forecast future points (k = N+1..N+numForecast) using day-by-day dates
  const forecast: GoldDataPoint[] = [];
  const lastDate = new Date(data[data.length - 1].date);
  for (let k = N + 1; k <= total; k++) {
    const x0_hat_k = Math.max(0, x1_hat[k - 1] - x1_hat[k - 2]);
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + (k - N));
    forecast.push({
      date: forecastDate.toISOString().split('T')[0],
      price: x0_hat_k,
      isPrediction: true
    } as any);
  }

  return forecast;
};

/**
 * Fit on train only, then predict the next test.length points aligned to test dates.
 */
export const runGM11TestPredict = (train: GoldDataPoint[], test: GoldDataPoint[]): GoldDataPoint[] => {
  const trainPrices = train.map(d => d.price);
  const Nt = trainPrices.length;
  const testSize = test.length;
  if (Nt < 3 || testSize < 1 || !isFinite(trainPrices[0])) return [];

  // AGO on train
  const x1: number[] = [trainPrices[0]];
  for (let i = 1; i < Nt; i++) x1.push(x1[i - 1] + trainPrices[i]);

  // Background values z (length Nt-1) and Y (x0 from 2..Nt)
  const z: number[] = [];
  for (let i = 0; i < Nt - 1; i++) z.push(0.5 * (x1[i] + x1[i + 1]));
  const y: number[] = trainPrices.slice(1);

  // Least squares for y = c*z + b => a = -c
  let sumZ = 0, sumY = 0, sumZZ = 0, sumZY = 0;
  for (let i = 0; i < Nt - 1; i++) {
    const zi = z[i], yi = y[i];
    sumZ += zi; sumY += yi; sumZZ += zi * zi; sumZY += zi * yi;
  }
  const n = Nt - 1;
  const denom = n * sumZZ - sumZ * sumZ;
  const EPS = 1e-12;
  if (Math.abs(denom) < EPS) return [];

  const c = (n * sumZY - sumZ * sumY) / denom;
  const b = (sumY - c * sumZ) / n;
  const a = -c;

  // x1_hat for k = 1..(Nt + testSize)
  const total = Nt + testSize;
  const x1_hat: number[] = new Array(total);
  if (Math.abs(a) < EPS) {
    for (let k = 1; k <= total; k++) x1_hat[k - 1] = trainPrices[0] + b * (k - 1);
  } else {
    const C = trainPrices[0] - b / a;
    for (let k = 1; k <= total; k++) x1_hat[k - 1] = C * Math.exp(-a * (k - 1)) + b / a;
  }

  // Return only the test window predictions aligned to test dates
  const result: GoldDataPoint[] = [];
  for (let i = 0; i < testSize; i++) {
    const k = Nt + 1 + i; // position in sequence for the i-th test point
    const x0_hat_k = Math.max(0, x1_hat[k - 1] - x1_hat[k - 2]);
    result.push({
      date: test[i].date,
      price: x0_hat_k,
      isPrediction: true
    } as any);
  }
  return result;
};
