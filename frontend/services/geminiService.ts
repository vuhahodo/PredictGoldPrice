
import { GoogleGenAI, Type } from "@google/genai";
import { GoldDataPoint, PredictionModel, PredictionResult } from "../types";

export const analyzeGoldPrices = async (
  data: GoldDataPoint[],
  modelType: PredictionModel
): Promise<PredictionResult> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  
  // Use a subset of data if it's too large to fit in prompt comfortably
  const recentData = data.slice(-50); 
  const dataString = recentData.map(d => `${d.date}: ${d.price}`).join(', ');

  const systemPrompt = `You are a financial data scientist specializing in gold market analysis. 
  The user is providing historical gold price data. 
  You need to simulate a ${modelType} prediction model.
  If LSTM is selected, act as a deep learning model trained on global gold price trends (similar to the Kaggle notebook logic).
  Return a structured JSON forecast for the next 10 business days.`;

  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: `Historical Data: ${dataString}. 
    Please predict the next 10 days of gold prices based on the ${modelType} architecture logic. 
    Focus on recent trends and volatility.`,
    config: {
      systemInstruction: systemPrompt,
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          forecast: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                date: { type: Type.STRING },
                price: { type: Type.NUMBER }
              },
              required: ["date", "price"]
            }
          },
          metrics: {
            type: Type.OBJECT,
            properties: {
              accuracy: { type: Type.NUMBER },
              trend: { type: Type.STRING, enum: ["up", "down", "neutral"] },
              confidence: { type: Type.NUMBER }
            },
            required: ["accuracy", "trend", "confidence"]
          }
        },
        required: ["forecast", "metrics"]
      }
    }
  });

  try {
    const result = JSON.parse(response.text);
    return {
      historical: data,
      forecast: result.forecast.map((f: any) => ({ ...f, isPrediction: true })),
      metrics: result.metrics
    };
  } catch (error) {
    console.error("Failed to parse AI response", error);
    throw new Error("AI prediction failed. Please try again.");
  }
};
