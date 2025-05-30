

import { TimeSeriesPoint, ForecastResult, ModelConfig } from '@/types/forecast';

const API_BASE_URL = '/api';

export const forecastAPI = {
async generateForecast(
    itemGroup: string,
    timeSeries: TimeSeriesPoint[],
    config: ModelConfig
): Promise<ForecastResult> {
    const requestBody = {
    itemGroup,
    timeSeries,
    modelType: config.modelType,
    frequency: config.frequency,
    periods: config.forecastPeriods,
    
      // SARIMA specific
    autoSelect: config.autoSelect,
    order: config.order,
    seasonalOrder: config.seasonalOrder,
    
      // Prophet specific
    prophetConfig: config.prophetConfig,
    
      // LSTM specific
    lstmConfig: config.lstmConfig
    };

    const response = await fetch(`${API_BASE_URL}/forecast`, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
},

async healthCheck(): Promise<{ status: string; message: string; models: string[] }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
}
};