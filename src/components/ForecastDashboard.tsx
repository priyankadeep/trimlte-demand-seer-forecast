import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import TimeSeriesChart from '@/components/TimeSeriesChart';
import ForecastChart from '@/components/ForecastChart';
import { ProcessedData, ModelConfig, ForecastResult } from '@/types/forecast';
import { TrendingUp, BarChart3 } from 'lucide-react';
import { forecastAPI } from '@/services/forecastAPI';
import { toast } from 'sonner';
interface ForecastDashboardProps {
  data: ProcessedData;
  config: ModelConfig;
  onForecastGenerated?: (itemGroup: string, forecast: ForecastResult) => void;
}

const ForecastDashboard: React.FC<ForecastDashboardProps> = ({ data, config, onForecastGenerated }) => {
  const [selectedItemGroup, setSelectedItemGroup] = useState<string>(data.itemGroups[0]?.name || '');
  const [forecasts, setForecasts] = useState<Map<string, ForecastResult>>(new Map());
  const [loading, setLoading] = useState(false);

  const selectedData = useMemo(() => {
    return data.itemGroups.find(ig => ig.name === selectedItemGroup);
  }, [data.itemGroups, selectedItemGroup]);

  // Simple forecast generation (mock implementation)
  const generateForecast = async (itemGroupName: string) => {
    setLoading(true);
    
    try {
      const itemGroup = data.itemGroups.find(ig => ig.name === itemGroupName);
      if (!itemGroup) {
        toast.error('Item group not found');
        return;
      }
  
      // Call the actual forecast API
      const result = await forecastAPI.generateForecast(
        itemGroupName,
        itemGroup.timeSeries,
        config
      );
  
      setForecasts(prev => new Map(prev).set(itemGroupName, result));
      
      // Call the callback to update parent component
      if (onForecastGenerated) {
        onForecastGenerated(itemGroupName, result);
      }
      
      toast.success(`Forecast generated for ${itemGroupName}`);
    } catch (error) {
      console.error('Error generating forecast:', error);
      toast.error('Failed to generate forecast. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
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
              <CardTitle className="text-sm font-medium text-slate-600">Time-Series Points (unique dates)</CardTitle>
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

