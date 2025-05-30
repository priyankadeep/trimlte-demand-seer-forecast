import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import TimeSeriesChart from '@/components/TimeSeriesChart';
import ForecastChart from '@/components/ForecastChart';
import { ProcessedData, ModelConfig, ForecastResult } from '@/types/forecast';
import { TrendingUp, BarChart3, Star, Info } from 'lucide-react';
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

  const generateForecast = async (itemGroupName: string) => {
    setLoading(true);
    
    try {
      const itemGroup = data.itemGroups.find(ig => ig.name === itemGroupName);
      if (!itemGroup) {
        toast.error('Item group not found');
        return;
      }
  
      // Call the forecast API with the selected model
      const result = await forecastAPI.generateForecast(
        itemGroupName,
        itemGroup.timeSeries,
        config
      );
  
      setForecasts(prev => new Map(prev).set(itemGroupName, result));
      
      if (onForecastGenerated) {
        onForecastGenerated(itemGroupName, result);
      }
      
      const modelName = config.modelType || 'SARIMA';
      toast.success(`${modelName} forecast generated for ${itemGroupName}`);
    } catch (error) {
      console.error('Error generating forecast:', error);
      toast.error('Failed to generate forecast. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  };

  const currentForecast = forecasts.get(selectedItemGroup);

  // Get model-specific badge color
  const getModelBadgeColor = (modelType: string) => {
    switch (modelType) {
      case 'SARIMA':
        return 'bg-blue-600';
      case 'Prophet':
        return 'bg-green-600';
      case 'LSTM':
        return 'bg-purple-600';
      default:
        return 'bg-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Model Recommendation Alert */}
      <Alert className="border-blue-200 bg-blue-50">
        <Star className="h-4 w-4 text-blue-600" />
        <AlertDescription className="text-blue-800">
          <strong>Recommended:</strong> SARIMA model provides the relatively better forecasts for demand planning. 
          Prophet and LSTM are experimental alternatives that may not perform as well for most use cases.
        </AlertDescription>
      </Alert>

      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-slate-800">Forecasting Dashboard</h2>
            <Badge className={`${getModelBadgeColor(config.modelType)} text-white`}>
              {config.modelType} Model
            </Badge>
            {config.modelType === 'SARIMA' && (
              <Badge variant="outline" className="border-blue-500 text-blue-600">
                <Star className="h-3 w-3 mr-1" />
                Recommended
              </Badge>
            )}
          </div>
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
            className={`${getModelBadgeColor(config.modelType)} hover:opacity-90`}
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Generating...
              </>
            ) : (
              <>
                <TrendingUp className="mr-2 h-4 w-4" />
                Generate {config.modelType} Forecast
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
              <CardTitle className="text-sm font-medium text-slate-600">Time-Series Points</CardTitle>
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
              {config.modelType} Forecast Results
              {currentForecast && (
                <Badge variant="secondary" className="ml-auto">
                  Generated ({currentForecast.metrics.method})
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
                Generate a {config.modelType} forecast to view predictions
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {currentForecast && (
        <Card className="bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Model Performance Metrics
              <Badge className={`${getModelBadgeColor(config.modelType)} text-white ml-2`}>
                {currentForecast.metrics.method?.toUpperCase()}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-6 gap-4">
              {currentForecast.metrics.mae !== undefined && (
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {currentForecast.metrics.mae.toFixed(2)}
                  </div>
                  <div className="text-sm text-slate-600">MAE</div>
                </div>
              )}
              
              {currentForecast.metrics.mse !== undefined && (
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-600">
                    {currentForecast.metrics.mse.toFixed(2)}
                  </div>
                  <div className="text-sm text-slate-600">MSE</div>
                </div>
              )}
              
              {currentForecast.metrics.rmse !== undefined && (
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {currentForecast.metrics.rmse.toFixed(2)}
                  </div>
                  <div className="text-sm text-slate-600">RMSE</div>
                </div>
              )}
              
              {currentForecast.metrics.aic !== undefined && (
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {currentForecast.metrics.aic.toFixed(0)}
                  </div>
                  <div className="text-sm text-slate-600">AIC</div>
                </div>
              )}

              {currentForecast.metrics.bic !== undefined && (
                <div className="text-center">
                  <div className="text-2xl font-bold text-indigo-600">
                    {currentForecast.metrics.bic.toFixed(0)}
                  </div>
                  <div className="text-sm text-slate-600">BIC</div>
                </div>
              )}
              
              {currentForecast.metrics.mape !== undefined && (
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {currentForecast.metrics.mape.toFixed(1)}%
                  </div>
                  <div className="text-sm text-slate-600">MAPE</div>
                </div>
              )}
            </div>
            
            {/* Model Configuration Summary - Clean display */}
            {currentForecast.model_params && (
              <div className="mt-6 p-4 bg-slate-50 rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <Info className="h-4 w-4 text-slate-600" />
                  <h4 className="font-medium text-slate-700">Model Configuration</h4>
                </div>
                
                {config.modelType === 'SARIMA' && currentForecast.model_params.order && (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-slate-600">Order (p,d,q): </span>
                      <span className="text-slate-800">
                        ({currentForecast.model_params.order.join(', ')})
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-slate-600">Seasonal Order: </span>
                      <span className="text-slate-800">
                        ({currentForecast.model_params.seasonal_order?.join(', ') || 'N/A'})
                      </span>
                    </div>
                  </div>
                )}
                
                {config.modelType === 'Prophet' && (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-slate-600">Growth: </span>
                      <span className="text-slate-800 capitalize">
                        {currentForecast.model_params.growth || 'Linear'}
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-slate-600">Seasonality: </span>
                      <span className="text-slate-800 capitalize">
                        {currentForecast.model_params.seasonality_mode || 'Additive'}
                      </span>
                    </div>
                  </div>
                )}
                
                {config.modelType === 'LSTM' && (
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-slate-600">Window Size: </span>
                      <span className="text-slate-800">
                        {currentForecast.model_params.window_size || config.lstmConfig.windowSize}
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-slate-600">Epochs: </span>
                      <span className="text-slate-800">
                        {currentForecast.model_params.epochs || config.lstmConfig.epochs}
                      </span>
                    </div>
                    <div>
                      <span className="font-medium text-slate-600">Architecture: </span>
                      <span className="text-slate-800">
                        {currentForecast.model_params.bidirectional ? 'Bidirectional' : 'Standard'} LSTM
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ForecastDashboard;