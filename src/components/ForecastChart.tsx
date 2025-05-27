
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from 'recharts';
import { TimeSeriesPoint } from '@/types/forecast';

interface ForecastChartProps {
  historicalData: TimeSeriesPoint[];
  forecastData: {
    date: string;
    mean: number;
    lower: number;
    upper: number;
  }[];
}

const ForecastChart: React.FC<ForecastChartProps> = ({ historicalData, forecastData }) => {
  // Combine historical and forecast data
  const historical = historicalData.slice(-24).map(point => ({
    date: new Date(point.date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short' 
    }),
    actual: point.value,
    type: 'historical'
  }));

  const forecast = forecastData.map(point => ({
    date: new Date(point.date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short' 
    }),
    forecast: point.mean,
    lower: point.lower,
    upper: point.upper,
    type: 'forecast'
  }));

  const combinedData = [...historical, ...forecast];

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={combinedData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis 
            dataKey="date" 
            stroke="#64748b"
            fontSize={12}
            tickMargin={8}
          />
          <YAxis 
            stroke="#64748b"
            fontSize={12}
            tickMargin={8}
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
          />
          
          {/* Confidence interval area */}
          <Area
            dataKey="upper"
            stroke="none"
            fill="#fecaca"
            fillOpacity={0.3}
          />
          <Area
            dataKey="lower"
            stroke="none"
            fill="#ffffff"
            fillOpacity={1}
          />
          
          {/* Historical data line */}
          <Line 
            type="monotone" 
            dataKey="actual" 
            stroke="#3b82f6" 
            strokeWidth={2}
            dot={{ fill: '#3b82f6', strokeWidth: 2, r: 3 }}
            connectNulls={false}
          />
          
          {/* Forecast line */}
          <Line 
            type="monotone" 
            dataKey="forecast" 
            stroke="#ef4444" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ fill: '#ef4444', strokeWidth: 2, r: 3 }}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;
