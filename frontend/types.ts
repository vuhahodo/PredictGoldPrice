
export enum PredictionModel {
  LSTM = 'LSTM',
  GM11 = 'GM(1,1)'
}

export interface GoldDataPoint {
  date: string;
  price: number;
  isPrediction?: boolean;
}

export interface PredictionResult {
  historical: GoldDataPoint[];
  forecast: GoldDataPoint[];
  testActual?: GoldDataPoint[];
  metrics: {
    accuracy: number;
    trend: 'up' | 'down' | 'neutral';
    confidence: number;
  };
}

export interface FileData {
  name: string;
  content: string;
  parsed: GoldDataPoint[];
}
