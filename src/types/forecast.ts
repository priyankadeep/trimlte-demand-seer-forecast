
export interface RawDataRow {
  'Order Date': string;
  'Item Groups': string;
  'Qty': number;
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

export interface ModelConfig {
  frequency: 'M' | 'W';
  forecastPeriods: number;
  autoSelect: boolean;
  order: [number, number, number];
  seasonalOrder: [number, number, number, number];
}

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
    rmse?: number;
    aic?: number;
  };
}
