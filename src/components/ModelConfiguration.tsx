
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { ModelConfig } from '@/types/forecast';
import React, { useEffect } from 'react';
interface ModelConfigurationProps {
  config: ModelConfig;
  onConfigChange: (config: ModelConfig) => void;
}

const ModelConfiguration: React.FC<ModelConfigurationProps> = ({ config, onConfigChange }) => {
  const updateConfig = (updates: Partial<ModelConfig>) => {
    onConfigChange({ ...config, ...updates });
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
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Basic Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
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

            <div className="flex items-center space-x-2">
              <Switch
                checked={config.autoSelect}
                onCheckedChange={(checked) => updateConfig({ autoSelect: checked })}
              />
              <Label>Auto-select parameters</Label>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">SARIMA Parameters</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
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
      </div>

      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="p-4">
          <div className="text-sm text-blue-800">
            <h4 className="font-medium mb-2">Parameter Guidelines:</h4>
            <ul className="list-disc list-inside space-y-1">
              <li><strong>p, d, q:</strong> Non-seasonal AR, differencing, and MA orders</li>
              <li><strong>P, D, Q:</strong> Seasonal AR, differencing, and MA orders</li>
              <li><strong>S:</strong> Seasonal period (12 for monthly, 52 for weekly)</li>
              <li><strong>Auto-select:</strong> Automatically finds optimal parameters using grid search</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelConfiguration;
