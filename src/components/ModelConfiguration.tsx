
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ModelConfig, ModelType } from '@/types/forecast';
import React, { useEffect } from 'react';

interface ModelConfigurationProps {
  config: ModelConfig;
  onConfigChange: (config: ModelConfig) => void;
}

const ModelConfiguration: React.FC<ModelConfigurationProps> = ({ config, onConfigChange }) => {
  const updateConfig = (updates: Partial<ModelConfig>) => {
    onConfigChange({ ...config, ...updates });
  };

  const updateProphetConfig = (updates: Partial<typeof config.prophetConfig>) => {
    onConfigChange({
      ...config,
      prophetConfig: { ...config.prophetConfig, ...updates }
    });
  };

  const updateLSTMConfig = (updates: Partial<typeof config.lstmConfig>) => {
    onConfigChange({
      ...config,
      lstmConfig: { ...config.lstmConfig, ...updates }
    });
  };

  // Auto-update seasonal period when frequency changes
  useEffect(() => {
    const seasonalPeriod = config.frequency === 'M' ? 12 : 52;
    if (config.seasonalOrder[3] !== seasonalPeriod) {
      updateConfig({
        seasonalOrder: [...config.seasonalOrder.slice(0, 3), seasonalPeriod] as [number, number, number, number]
      });
    }
  }, [config.frequency]);

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Model Selection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Forecasting Model</Label>
            <Select 
              value={config.modelType} 
              onValueChange={(value: ModelType) => updateConfig({ modelType: value })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="SARIMA">SARIMA</SelectItem>
                <SelectItem value="Prophet">Prophet</SelectItem>
                <SelectItem value="LSTM">LSTM</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Frequency</Label>
              <Select value={config.frequency} onValueChange={(value: 'M' | 'W') => updateConfig({ frequency: value })}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="M">Monthly</SelectItem>
                  <SelectItem value="W">Weekly</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Forecast Periods</Label>
              <Input
                type="number"
                value={config.forecastPeriods}
                onChange={(e) => updateConfig({ forecastPeriods: parseInt(e.target.value) || 12 })}
                min="1"
                max="36"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model-specific configurations */}
      <Tabs value={config.modelType} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="SARIMA">SARIMA Config</TabsTrigger>
          <TabsTrigger value="Prophet">Prophet Config</TabsTrigger>
          <TabsTrigger value="LSTM">LSTM Config</TabsTrigger>
        </TabsList>

        {/* SARIMA Configuration */}
        <TabsContent value="SARIMA" className="space-y-4">
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">SARIMA Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={config.autoSelect}
                    onCheckedChange={(checked) => updateConfig({ autoSelect: checked })}
                  />
                  <Label>Auto-select parameters</Label>
                </div>

                <div className="space-y-3">
                  <Label>Order (p, d, q)</Label>
                  <div className="grid grid-cols-3 gap-2">
                    <Input
                      type="number"
                      value={config.order[0]}
                      onChange={(e) => updateConfig({ 
                        order: [parseInt(e.target.value) || 0, config.order[1], config.order[2]] 
                      })}
                      min="0"
                      max="5"
                      placeholder="p"
                      disabled={config.autoSelect}
                    />
                    <Input
                      type="number"
                      value={config.order[1]}
                      onChange={(e) => updateConfig({ 
                        order: [config.order[0], parseInt(e.target.value) || 0, config.order[2]] 
                      })}
                      min="0"
                      max="2"
                      placeholder="d"
                      disabled={config.autoSelect}
                    />
                    <Input
                      type="number"
                      value={config.order[2]}
                      onChange={(e) => updateConfig({ 
                        order: [config.order[0], config.order[1], parseInt(e.target.value) || 0] 
                      })}
                      min="0"
                      max="5"
                      placeholder="q"
                      disabled={config.autoSelect}
                    />
                  </div>
                </div>

                <div className="space-y-3">
                  <Label>Seasonal Order (P, D, Q, S)</Label>
                  <div className="grid grid-cols-4 gap-2">
                    <Input
                      type="number"
                      value={config.seasonalOrder[0]}
                      onChange={(e) => updateConfig({ 
                        seasonalOrder: [parseInt(e.target.value) || 0, config.seasonalOrder[1], config.seasonalOrder[2], config.seasonalOrder[3]] 
                      })}
                      min="0"
                      max="3"
                      placeholder="P"
                      disabled={config.autoSelect}
                    />
                    <Input
                      type="number"
                      value={config.seasonalOrder[1]}
                      onChange={(e) => updateConfig({ 
                        seasonalOrder: [config.seasonalOrder[0], parseInt(e.target.value) || 0, config.seasonalOrder[2], config.seasonalOrder[3]] 
                      })}
                      min="0"
                      max="2"
                      placeholder="D"
                      disabled={config.autoSelect}
                    />
                    <Input
                      type="number"
                      value={config.seasonalOrder[2]}
                      onChange={(e) => updateConfig({ 
                        seasonalOrder: [config.seasonalOrder[0], config.seasonalOrder[1], parseInt(e.target.value) || 0, config.seasonalOrder[3]] 
                      })}
                      min="0"
                      max="3"
                      placeholder="Q"
                      disabled={config.autoSelect}
                    />
                    <Input
                      type="number"
                      value={config.seasonalOrder[3]}
                      onChange={(e) => updateConfig({ 
                        seasonalOrder: [config.seasonalOrder[0], config.seasonalOrder[1], config.seasonalOrder[2], parseInt(e.target.value) || 12] 
                      })}
                      min="1"
                      placeholder="S"
                      disabled={config.autoSelect}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-blue-50 border-blue-200">
              <CardContent className="p-4">
                <div className="text-sm text-blue-800">
                  <h4 className="font-medium mb-2">SARIMA Guidelines:</h4>
                  <ul className="list-disc list-inside space-y-1">
                    <li><strong>p, d, q:</strong> Non-seasonal AR, differencing, and MA orders</li>
                    <li><strong>P, D, Q:</strong> Seasonal AR, differencing, and MA orders</li>
                    <li><strong>S:</strong> Seasonal period (12 for monthly, 52 for weekly)</li>
                    <li><strong>Auto-select:</strong> Automatically finds optimal parameters</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Prophet Configuration */}
        <TabsContent value="Prophet" className="space-y-4">
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Prophet Core Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Growth Model</Label>
                  <Select 
                    value={config.prophetConfig.growth} 
                    onValueChange={(value: 'linear' | 'logistic') => updateProphetConfig({ growth: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="linear">Linear</SelectItem>
                      <SelectItem value="logistic">Logistic</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Seasonality Mode</Label>
                  <Select 
                    value={config.prophetConfig.seasonalityMode} 
                    onValueChange={(value: 'additive' | 'multiplicative') => updateProphetConfig({ seasonalityMode: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="additive">Additive</SelectItem>
                      <SelectItem value="multiplicative">Multiplicative</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Change Point Range: {config.prophetConfig.changePointRange}</Label>
                  <Slider
                    value={[config.prophetConfig.changePointRange]}
                    onValueChange={([value]) => updateProphetConfig({ changePointRange: value })}
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Change Point Prior Scale: {config.prophetConfig.changePointPriorScale}</Label>
                  <Slider
                    value={[config.prophetConfig.changePointPriorScale]}
                    onValueChange={([value]) => updateProphetConfig({ changePointPriorScale: value })}
                    min={0.01}
                    max={1.0}
                    step={0.01}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Prophet Seasonality</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={config.prophetConfig.yearlySeasonality}
                    onCheckedChange={(checked) => updateProphetConfig({ yearlySeasonality: checked })}
                  />
                  <Label>Yearly Seasonality</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    checked={config.prophetConfig.weeklySeasonality}
                    onCheckedChange={(checked) => updateProphetConfig({ weeklySeasonality: checked })}
                  />
                  <Label>Weekly Seasonality</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    checked={config.prophetConfig.dailySeasonality}
                    onCheckedChange={(checked) => updateProphetConfig({ dailySeasonality: checked })}
                  />
                  <Label>Daily Seasonality</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    checked={config.prophetConfig.addQuarterly}
                    onCheckedChange={(checked) => updateProphetConfig({ addQuarterly: checked })}
                  />
                  <Label>Add Quarterly Seasonality</Label>
                </div>

                {config.prophetConfig.addQuarterly && (
                  <div className="space-y-2">
                    <Label>Quarterly Fourier Order</Label>
                    <Input
                      type="number"
                      value={config.prophetConfig.quarterlyFourierOrder}
                      onChange={(e) => updateProphetConfig({ quarterlyFourierOrder: parseInt(e.target.value) || 5 })}
                      min="1"
                      max="10"
                    />
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <Card className="bg-green-50 border-green-200">
            <CardContent className="p-4">
              <div className="text-sm text-green-800">
                <h4 className="font-medium mb-2">Prophet Guidelines:</h4>
                <ul className="list-disc list-inside space-y-1">
                  <li><strong>Linear Growth:</strong> For trends that continue indefinitely</li>
                  <li><strong>Logistic Growth:</strong> For trends that level off at a carrying capacity</li>
                  <li><strong>Additive:</strong> Seasonal effects are constant over time</li>
                  <li><strong>Multiplicative:</strong> Seasonal effects scale with the trend</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* LSTM Configuration */}
        <TabsContent value="LSTM" className="space-y-4">
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">LSTM Architecture</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Window Size (Look-back periods)</Label>
                  <Input
                    type="number"
                    value={config.lstmConfig.windowSize}
                    onChange={(e) => updateLSTMConfig({ windowSize: parseInt(e.target.value) || 12 })}
                    min="3"
                    max="60"
                  />
                </div>

                <div className="space-y-2">
                  <Label>First LSTM Layer Units</Label>
                  <Input
                    type="number"
                    value={config.lstmConfig.lstmUnits1}
                    onChange={(e) => updateLSTMConfig({ lstmUnits1: parseInt(e.target.value) || 64 })}
                    min="8"
                    max="256"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Second LSTM Layer Units</Label>
                  <Input
                    type="number"
                    value={config.lstmConfig.lstmUnits2}
                    onChange={(e) => updateLSTMConfig({ lstmUnits2: parseInt(e.target.value) || 32 })}
                    min="8"
                    max="128"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Dropout Rate: {config.lstmConfig.dropoutRate}</Label>
                  <Slider
                    value={[config.lstmConfig.dropoutRate]}
                    onValueChange={([value]) => updateLSTMConfig({ dropoutRate: value })}
                    min={0.0}
                    max={0.8}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    checked={config.lstmConfig.bidirectional}
                    onCheckedChange={(checked) => updateLSTMConfig({ bidirectional: checked })}
                  />
                  <Label>Bidirectional LSTM</Label>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">LSTM Training</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Epochs</Label>
                  <Input
                    type="number"
                    value={config.lstmConfig.epochs}
                    onChange={(e) => updateLSTMConfig({ epochs: parseInt(e.target.value) || 50 })}
                    min="10"
                    max="200"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Batch Size</Label>
                  <Select 
                    value={config.lstmConfig.batchSize.toString()} 
                    onValueChange={(value) => updateLSTMConfig({ batchSize: parseInt(value) })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="8">8</SelectItem>
                      <SelectItem value="16">16</SelectItem>
                      <SelectItem value="32">32</SelectItem>
                      <SelectItem value="64">64</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Validation Split: {config.lstmConfig.validationSplit}</Label>
                  <Slider
                    value={[config.lstmConfig.validationSplit]}
                    onValueChange={([value]) => updateLSTMConfig({ validationSplit: value })}
                    min={0.1}
                    max={0.3}
                    step={0.05}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-purple-50 border-purple-200">
            <CardContent className="p-4">
              <div className="text-sm text-purple-800">
                <h4 className="font-medium mb-2">LSTM Guidelines:</h4>
                <ul className="list-disc list-inside space-y-1">
                  <li><strong>Window Size:</strong> Number of past periods to use for prediction</li>
                  <li><strong>LSTM Units:</strong> Higher values = more complex patterns but slower training</li>
                  <li><strong>Dropout:</strong> Prevents overfitting, use 0.2-0.5 for most cases</li>
                  <li><strong>Bidirectional:</strong> Processes data forward and backward (better accuracy)</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ModelConfiguration;
