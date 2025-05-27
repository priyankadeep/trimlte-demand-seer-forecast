
import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import TimeSeriesChart from '@/components/TimeSeriesChart';
import ForecastChart from '@/components/ForecastChart';
import { ProcessedData, ModelConfig, ForecastResult } from '@/types/forecast';
import { TrendingUp, BarChart3 } from 'lucide-react';

interface ForecastDashboardProps {
  data: ProcessedData;
  config: ModelConfig;
}

const ForecastDashboard: React.FC<ForecastDashboardProps> = ({ data, config }) => {
  const [selectedItemGroup, setSelectedItemGroup] = useState<string>(data.itemGroups[0]?.name || '');
  const [forecasts, setForecasts] = useState<Map<string, ForecastResult>>(new Map());
  const [loading, setLoading] = useState(false);

  const selectedData = useMemo(() => {
    return data.itemGroups.find(ig => ig.name === selectedItemGroup);
  }, [data.itemGroups, selectedItemGroup]);

  // Simple forecast generation (mock implementation)
  const generateForecast = async (itemGroupName: string) => {
    setLoading(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const itemGroup = data.itemGroups.find(ig => ig.name === itemGroupName);
    if (!itemGroup) return;

    // Generate mock forecast data
    const lastPoint = itemGroup.timeSeries[itemGroup.timeSeries.length - 1];
    const avgValue = itemGroup.stats.avgQuantity;
    const trend = avgValue * 0.05; // 5% growth trend
    
    const forecast = Array.from({ length: config.forecastPeriods }, (_, i) => {
      const baseValue = avgValue + (trend * i);
      const noise = (Math.random() - 0.5) * avgValue * 0.2;
      const seasonal = Math.sin((i * 2 * Math.PI) / 12) * avgValue * 0.1;
      
      const mean = Math.max(0, baseValue + noise + seasonal);
      const confidence = mean * 0.2;
      
      const forecastDate = new Date(lastPoint.date);
      forecastDate.setMonth(forecastDate.getMonth() + i + 1);
      
      return {
        date: forecastDate.toISOString().split('T')[0],
        mean,
        lower: Math.max(0, mean - confidence),
        upper: mean + confidence
      };
    });

    const result: ForecastResult = {
      itemGroup: itemGroupName,
      forecast,
      metrics: {
        mae: Math.random() * 50 + 10,
        rmse: Math.random() * 70 + 20,
        aic: Math.random() * 1000 + 500
      }
    };

    setForecasts(prev => new Map(prev).set(itemGroupName, result));
    setLoading(false);
  };

  const currentForecast = forecasts.get(selectedItemGroup);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-bold text-slate-800">Forecasting Dashboard</h2>
          <p className="text-slate-600">
            {data.itemGroups.length} item groups • {config.forecastPeriods} period forecast
          </p>
        </div>
        
        <div className="flex gap-3">
          <Select value={selectedItemGroup} onValueChange={setSelectedItemGroup}>
            <SelectTrigger className="w-64">
              <SelectValue placeholder="Select item group" />
            </SelectTrigger>
            <SelectContent>
              {data.itemGroups.map(ig => (
                <SelectItem key={ig.name} value={ig.name}>
                  {ig.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Button 
            onClick={() => generateForecast(selectedItemGroup)}
            disabled={loading || !selectedItemGroup}
            className="bg-green-600 hover:bg-green-700"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Generating...
              </>
            ) : (
              <>
                <TrendingUp className="mr-2 h-4 w-4" />
                Generate Forecast
              </>
            )}
          </Button>
        </div>
      </div>

      {selectedData && (
        <div className="grid lg:grid-cols-4 gap-6">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Total Quantity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-slate-900">
                {selectedData.stats.totalQuantity.toLocaleString()}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Total Revenue</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-slate-900">
                ${selectedData.stats.totalRevenue.toLocaleString()}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Avg Quantity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-slate-900">
                {selectedData.stats.avgQuantity.toFixed(1)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Data Points</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-slate-900">
                {selectedData.timeSeries.length}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Historical Data
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedData && (
              <TimeSeriesChart data={selectedData.timeSeries} />
            )}
          </CardContent>
        </Card>

        <Card className="bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Forecast Results
              {currentForecast && (
                <Badge variant="secondary" className="ml-auto">
                  Generated
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {currentForecast && selectedData ? (
              <ForecastChart 
                historicalData={selectedData.timeSeries}
                forecastData={currentForecast.forecast}
              />
            ) : (
              <div className="h-64 flex items-center justify-center text-slate-500">
                Generate a forecast to view predictions
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {currentForecast && (
        <Card className="bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <CardTitle>Model Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {currentForecast.metrics.mae?.toFixed(2)}
                </div>
                <div className="text-sm text-slate-600">Mean Absolute Error</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {currentForecast.metrics.rmse?.toFixed(2)}
                </div>
                <div className="text-sm text-slate-600">Root Mean Square Error</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {currentForecast.metrics.aic?.toFixed(0)}
                </div>
                <div className="text-sm text-slate-600">Akaike Information Criterion</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ForecastDashboard;
