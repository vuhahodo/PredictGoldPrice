
import { GoldDataPoint } from '../types';

/**
 * Basic GM(1,1) Implementation for Time Series Prediction
 */
export const runGM11 = (data: GoldDataPoint[], forecastSteps: number = 7): GoldDataPoint[] => {
  const x0 = data.map(d => d.price);
  const n = x0.length;
  if (n < 4) return [];

  // 1-AGO (Accumulating Generation Operator)
  const x1 = new Array(n);
  x1[0] = x0[0];
  for (let i = 1; i < n; i++) {
    x1[i] = x1[i - 1] + x0[i];
  }

  // Mean sequence generation
  const z1 = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    z1[i] = 0.5 * (x1[i] + x1[i + 1]);
  }

  // Least squares to find parameters a (development coefficient) and b (grey action quantity)
  // B = [-z1(2) 1; -z1(3) 1; ...; -z1(n) 1]
  // Y = [x0(2); x0(3); ...; x0(n)]
  let sumZ1Squared = 0;
  let sumZ1 = 0;
  let sumX0 = 0;
  let sumZ1X0 = 0;

  for (let i = 0; i < n - 1; i++) {
    sumZ1Squared += z1[i] * z1[i];
    sumZ1 += z1[i];
    sumX0 += x0[i + 1];
    sumZ1X0 += z1[i] * x0[i + 1];
  }

  const denominator = (n - 1) * sumZ1Squared - sumZ1 * sumZ1;
  const a = (sumZ1 * sumX0 - (n - 1) * sumZ1X0) / denominator;
  const b = (sumZ1Squared * sumX0 - sumZ1 * sumZ1X0) / denominator;

  // Prediction formula: x1_hat(k+1) = (x0(1) - b/a) * exp(-a*k) + b/a
  const forecast: GoldDataPoint[] = [];
  const lastDate = new Date(data[n - 1].date);

  for (let k = 1; k <= forecastSteps; k++) {
    const t = n + k - 1;
    const x1_hat_curr = (x0[0] - b / a) * Math.exp(-a * t) + b / a;
    const x1_hat_prev = (x0[0] - b / a) * Math.exp(-a * (t - 1)) + b / a;
    const predictedPrice = x1_hat_curr - x1_hat_prev;

    const nextDate = new Date(lastDate);
    nextDate.setDate(lastDate.getDate() + k);

    forecast.push({
      date: nextDate.toISOString().split('T')[0],
      price: Math.max(0, predictedPrice),
      isPrediction: true
    });
  }

  return forecast;
};
