import { ProcessedData, ModelConfig, ForecastResult } from '@/types/forecast';

const API_BASE_URL = 'http://localhost:5001';
export const forecastAPI = {
async generateForecast(
    itemGroup: string,
    timeSeries: any[],
    config: ModelConfig
): Promise<ForecastResult> {
    try {
    const response = await fetch(`${API_BASE_URL}/api/forecast`, {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify({
        itemGroup,
        timeSeries,
        frequency: config.frequency,
        periods: config.forecastPeriods,
        order: config.order,
        seasonalOrder: config.seasonalOrder,
        autoSelect: config.autoSelect
        }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
    } catch (error) {
    console.error('Error calling forecast API:', error);
    throw error;
    }
},

async batchForecast(
    data: ProcessedData,
    config: ModelConfig
): Promise<Map<string, ForecastResult>> {
    try {
    const response = await fetch(`${API_BASE_URL}/api/forecast/batch`, {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify({
        itemGroups: data.itemGroups,
        config
        }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const results = await response.json();
    return new Map(Object.entries(results));
    } catch (error) {
    console.error('Error calling batch forecast API:', error);
    throw error;
    }
}
};