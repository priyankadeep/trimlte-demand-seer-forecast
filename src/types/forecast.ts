
export interface RawDataRow {
  'Order Date': string;
  'Item Groups': string;
  'Qty': number;
  'Price': number;  // Added for correct revenue calculation
  'Qty - w/Sub-Total': number;  // Added for correct revenue calculation
  'Ext. Price': number;
  'Tran #': string;
}

export interface TimeSeriesPoint {
  date: string;
  value: number;
  revenue?: number;
  transactions?: number;
}

export interface ItemGroupData {
  name: string;
  timeSeries: TimeSeriesPoint[];
  stats: {
    totalQuantity: number;
    totalRevenue: number;
    avgQuantity: number;
    maxQuantity: number;
    minQuantity: number;
  };
}

export interface ProcessedData {
  itemGroups: ItemGroupData[];
  dateRange: {
    start: string;
    end: string;
  };
  totalRecords: number;
}

// Add model type enum
export type ModelType = 'SARIMA' | 'Prophet' | 'LSTM';

// Updated ModelConfig interface with all three models
export interface ModelConfig {
  // Common settings
  frequency: 'M' | 'W';
  forecastPeriods: number;
  modelType: ModelType;
  
  // SARIMA specific
  autoSelect: boolean;
  order: [number, number, number];
  seasonalOrder: [number, number, number, number];
  
  // Prophet specific
  prophetConfig: {
    growth: 'linear' | 'logistic';
    seasonalityMode: 'additive' | 'multiplicative';
    changePointRange: number;
    changePointPriorScale: number;
    yearlySeasonality: boolean;
    weeklySeasonality: boolean;
    dailySeasonality: boolean;
    addQuarterly: boolean;
    quarterlyFourierOrder: number;
  };
  
  // LSTM specific
  lstmConfig: {
    windowSize: number;
    epochs: number;
    batchSize: number;
    lstmUnits1: number;
    lstmUnits2: number;
    dropoutRate: number;
    bidirectional: boolean;
    validationSplit: number;
  };
}

// Updated ForecastResult interface with enhanced metrics
export interface ForecastResult {
  itemGroup: string;
  forecast: {
    date: string;
    mean: number;
    lower: number;
    upper: number;
  }[];
  metrics: {
    mae?: number;
    mse?: number;  // Added MSE
    rmse?: number;
    mape?: number;
    aic?: number;
    bic?: number;
    direction_accuracy?: number;
    method: string;
  };
  diagnostics?: {
    ljung_box_pvalue?: number;
    jarque_bera_pvalue?: number;
    heteroscedasticity_pvalue?: number;
  };
  model_params?: any;
  modelType?: ModelType;
  warning?: string;
}


