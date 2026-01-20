"""
MCM Problem B: Elevator Traffic Analysis and NHPP Modeling
Robust Version - Handles Missing Values and Large Datasets
Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import gc
import re

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ElevatorDataAnalyzer:
    """Main class for elevator data analysis and NHPP modeling"""

    def __init__(self, data_folder='./data'):
        """Initialize the analyzer with data folder path"""
        self.data_folder = data_folder
        self.hall_calls = None
        self.car_calls = None
        self.car_stops = None
        self.car_departures = None
        self.load_changes = None
        self.maintenance_mode = None
        self.day_types = None

    def _parse_datetime_flexible(self, datetime_str):
        """Robust datetime parser for various formats"""
        if pd.isna(datetime_str):
            return None

        try:
            # Clean up the string
            datetime_str = str(datetime_str).strip()

            # Replace multiple spaces with single space
            datetime_str = re.sub(r'\s+', ' ', datetime_str)

            # Split into date and time parts
            parts = datetime_str.split(' ')
            if len(parts) < 2:
                return None

            date_part = parts[0]
            time_part = parts[1]

            # Parse date (handle YYYY/MM/DD or YYYY/M/D)
            date_parts = date_part.split('/')
            year = int(date_parts[0])
            month = int(date_parts[1])
            day = int(date_parts[2]) if len(date_parts) > 2 else 1

            # Parse time (handle H:MM:SS or HH:MM:SS)
            time_parts = time_part.split(':')
            hour = int(time_parts[0]) if time_parts[0] else 0
            minute = int(time_parts[1]) if len(time_parts) > 1 and time_parts[1] else 0
            second = int(time_parts[2]) if len(time_parts) > 2 and time_parts[2] else 0

            return datetime(year, month, day, hour, minute, second)

        except (ValueError, IndexError, AttributeError) as e:
            print(f"Warning: Could not parse datetime: '{datetime_str}' - {e}")
            return None

    def load_data_safely(self):
        """Load all CSV files safely, handling missing values"""
        print("Loading data files with safe parsing...")

        try:
            # Define safer data types - use object first, then convert
            dtype_safe = {
                'Time': str,
                'Elevator ID': str,
                'Direction': str,
                'Floor': str,  # Load as string first
                'Load In (kg)': str,
                'Load Out (kg)': str,
                'Action': str,
                'Mode': str
            }

            # Load each file safely
            self.hall_calls = self._load_single_file('hall_calls.csv',
                                                   usecols=['Time', 'Elevator ID', 'Direction', 'Floor'])

            self.car_calls = self._load_single_file('car_calls.csv',
                                                  usecols=['Time', 'Elevator ID', 'Floor', 'Action'])

            self.car_stops = self._load_single_file('car_stops.csv',
                                                  usecols=['Time', 'Elevator ID', 'Floor', 'Direction'])

            self.car_departures = self._load_single_file('car_departures.csv',
                                                       usecols=['Time', 'Elevator ID', 'Floor'])

            self.load_changes = self._load_single_file('load_changes.csv',
                                                     usecols=['Time', 'Elevator ID', 'Floor',
                                                              'Load In (kg)', 'Load Out (kg)'])

            self.maintenance_mode = self._load_single_file('maintenance_mode.csv',
                                                         usecols=['Time', 'Elevator ID', 'Mode', 'Action'])

            print("\nData loaded successfully!")
            print(f"hall_calls shape: {self.hall_calls.shape}")
            print(f"car_calls shape: {self.car_calls.shape}")
            print(f"car_stops shape: {self.car_stops.shape}")

            # Show unique elevator IDs
            if self.hall_calls is not None:
                unique_elevators = self.hall_calls['Elevator ID'].dropna().unique()
                print(f"\nUnique elevator IDs in hall_calls: {sorted(unique_elevators)}")

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    def _load_single_file(self, filename, usecols):
        """Load a single CSV file safely"""
        filepath = os.path.join(self.data_folder, filename)
        print(f"  Loading {filename}...")

        try:
            # Read the file
            df = pd.read_csv(filepath, usecols=usecols, dtype=str, low_memory=False)
            print(f"    Read {len(df):,} rows")

            # Parse datetime
            print(f"    Parsing datetime...")
            df['Time'] = df['Time'].apply(self._parse_datetime_flexible)

            # Drop rows with invalid datetime
            initial_count = len(df)
            df = df[df['Time'].notna()]
            print(f"    Kept {len(df):,} rows after datetime parsing (dropped {initial_count - len(df)})")

            # Convert other columns safely
            for col in df.columns:
                if col == 'Time':
                    continue

                # Handle numeric columns
                if col in ['Floor', 'Load In (kg)', 'Load Out (kg)']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # For Floor, fill NaN with -1 (to indicate unknown)
                    if col == 'Floor':
                        df[col] = df[col].fillna(-1).astype(int)
                        # Replace negative values with 1 (ground floor)
                        df.loc[df[col] < 0, col] = 1
                    else:
                        df[col] = df[col].fillna(0).astype(float)

                # Handle categorical columns
                elif col in ['Elevator ID', 'Direction', 'Action', 'Mode']:
                    df[col] = df[col].fillna('Unknown').astype('category')

            print(f"    Data types: {dict(df.dtypes)}")

            return df

        except Exception as e:
            print(f"    Error loading {filename}: {e}")
            return None

    def preprocess_data(self):
        """Preprocess the data"""
        print("\nPreprocessing data...")

        dataframes = {
            'hall_calls': self.hall_calls,
            'car_calls': self.car_calls,
            'car_stops': self.car_stops,
            'load_changes': self.load_changes
        }

        for name, df in dataframes.items():
            if df is not None and len(df) > 0:
                print(f"  Processing {name}...")

                # Extract date and time components
                df['Date'] = df['Time'].dt.date
                df['Hour'] = df['Time'].dt.hour.astype('int8')
                df['Minute'] = df['Time'].dt.minute.astype('int8')
                df['Second'] = df['Time'].dt.second.astype('int8')

                # Create 5-minute time slices
                df['5min_slice'] = ((df['Hour'] * 60 + df['Minute']) // 5).astype('int16')

                # Extract day of week
                df['DayOfWeek'] = df['Time'].dt.dayofweek.astype('int8')
                df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype('bool')

                print(f"    Added time features, shape: {df.shape}")

        print("Data preprocessing completed.")

    def analyze_traffic_patterns(self):
        """Analyze traffic patterns for day type classification"""
        print("\nAnalyzing traffic patterns for day type classification...")

        if self.hall_calls is None or len(self.hall_calls) == 0:
            print("Error: No hall_calls data available!")
            return None

        # Extract daily features
        daily_features = self._extract_daily_features()

        if len(daily_features) < 2:
            print("Error: Not enough days for classification!")
            return None

        # Cluster days
        day_types = self._cluster_days(daily_features)

        # Analyze characteristics
        self._analyze_cluster_characteristics(daily_features, day_types)

        self.day_types = day_types
        return day_types

    def _extract_daily_features(self):
        """Extract features for each day"""

        # Get unique dates
        dates = sorted(self.hall_calls['Date'].unique())
        print(f"Found {len(dates)} unique days")

        features_list = []

        for i, date in enumerate(dates):
            # Filter data for this date
            date_mask = self.hall_calls['Date'] == date
            date_data = self.hall_calls[date_mask]

            if len(date_data) == 0:
                continue

            # Calculate basic features
            total_calls = len(date_data)

            # Time distribution features
            hours = date_data['Hour'].values
            morning_peak = np.sum((hours >= 8) & (hours < 10)) / total_calls if total_calls > 0 else 0
            evening_peak = np.sum((hours >= 17) & (hours < 19)) / total_calls if total_calls > 0 else 0
            lunch_peak = np.sum((hours >= 12) & (hours < 14)) / total_calls if total_calls > 0 else 0
            night_ratio = np.sum((hours >= 20) | (hours < 6)) / total_calls if total_calls > 0 else 0

            # Direction features
            if 'Direction' in date_data.columns:
                up_ratio = (date_data['Direction'] == 'Up').sum() / total_calls if total_calls > 0 else 0
                down_ratio = (date_data['Direction'] == 'Down').sum() / total_calls if total_calls > 0 else 0
            else:
                up_ratio = down_ratio = 0.5

            # Calculate hourly distribution for peakiness
            hourly_counts = date_data.groupby('Hour').size()
            if len(hourly_counts) > 1:
                probs = hourly_counts.values / hourly_counts.values.sum()
                peakiness = stats.kurtosis(probs, fisher=False)

                # Calculate uniformity (entropy)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(probs))
                uniformity = entropy / max_entropy if max_entropy > 0 else 0
            else:
                peakiness = 0
                uniformity = 0

            features = {
                'Date': date,
                'TotalCalls': total_calls,
                'MorningPeak_8_10': morning_peak,
                'EveningPeak_17_19': evening_peak,
                'LunchPeak_12_14': lunch_peak,
                'NightRatio_20_6': night_ratio,
                'UpRatio': up_ratio,
                'DownRatio': down_ratio,
                'Peakiness': peakiness,
                'Uniformity': uniformity
            }

            features_list.append(features)

            if i % 10 == 0 and i > 0:
                print(f"    Processed {i+1}/{len(dates)} days")

        return pd.DataFrame(features_list)

    def _cluster_days(self, daily_features):
        """Cluster days into Weekday and Holiday patterns"""

        # Select features for clustering
        feature_cols = ['MorningPeak_8_10', 'EveningPeak_17_19',
                       'LunchPeak_12_14', 'NightRatio_20_6',
                       'UpRatio', 'Peakiness', 'Uniformity']

        # Check if we have enough data
        if len(daily_features) < 2:
            print("Not enough days for clustering!")
            return daily_features[['Date']].copy()

        X = daily_features[feature_cols].values

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-means
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Calculate silhouette score
        if len(np.unique(clusters)) > 1:
            silhouette_avg = silhouette_score(X_scaled, clusters)
            print(f"Clustering silhouette score: {silhouette_avg:.3f}")
        else:
            print("Warning: Only one cluster found!")

        # Assign clusters
        daily_features['Cluster'] = clusters

        # Determine which cluster is Weekday vs Holiday
        # Based on morning peak intensity (weekdays should have higher morning peaks)
        cluster_stats = []
        for cluster_id in [0, 1]:
            cluster_data = daily_features[daily_features['Cluster'] == cluster_id]
            stats_dict = {
                'Cluster': cluster_id,
                'MorningPeak_Mean': cluster_data['MorningPeak_8_10'].mean(),
                'Size': len(cluster_data)
            }
            cluster_stats.append(stats_dict)

        cluster_stats_df = pd.DataFrame(cluster_stats)
        weekday_cluster = cluster_stats_df.loc[cluster_stats_df['MorningPeak_Mean'].idxmax(), 'Cluster']

        # Assign day types
        daily_features['DayType'] = daily_features['Cluster'].apply(
            lambda x: 'Weekday' if x == weekday_cluster else 'Holiday'
        )

        return daily_features[['Date', 'DayType', 'Cluster']]

    def _analyze_cluster_characteristics(self, daily_features, day_types):
        """Analyze cluster characteristics"""

        print("\n" + "="*60)
        print("DAY TYPE CHARACTERISTICS ANALYSIS")
        print("="*60)

        # Merge features with day types
        merged_df = pd.merge(daily_features, day_types, on='Date')

        for day_type in ['Weekday', 'Holiday']:
            type_data = merged_df[merged_df['DayType'] == day_type]

            if len(type_data) == 0:
                print(f"\nNo {day_type} data found!")
                continue

            print(f"\n{day_type} Characteristics ({len(type_data)} days):")
            print("-" * 40)

            metrics = {
                'Morning Peak (8-10) Ratio': type_data['MorningPeak_8_10'].mean(),
                'Evening Peak (17-19) Ratio': type_data['EveningPeak_17_19'].mean(),
                'Lunch Peak (12-14) Ratio': type_data['LunchPeak_12_14'].mean(),
                'Night Activity (20-6) Ratio': type_data['NightRatio_20_6'].mean(),
                'Up/Down Ratio': type_data['UpRatio'].mean(),
                'Peakiness (Kurtosis)': type_data['Peakiness'].mean(),
                'Uniformity (Entropy)': type_data['Uniformity'].mean(),
                'Avg Daily Calls': type_data['TotalCalls'].mean(),
            }

            for metric, value in metrics.items():
                print(f"  {metric:30}: {value:.3f}")

    def fit_nhpp_models(self, train_days=20):
        """Fit NHPP models for Weekday and Holiday patterns"""
        print("\n" + "="*60)
        print("FITTING NON-HOMOGENEOUS POISSON PROCESS MODELS")
        print("="*60)

        if self.day_types is None:
            print("Error: Day types not classified yet!")
            return None, None

        # Get all dates
        all_dates = sorted(self.hall_calls['Date'].unique())

        if len(all_dates) < 30:
            print(f"Warning: Only {len(all_dates)} days of data available.")
            train_days = min(train_days, len(all_dates) - 10)

        train_dates = all_dates[:train_days]
        val_dates = all_dates[train_days:train_days+10] if len(all_dates) > train_days else []

        print(f"Training on {len(train_dates)} days: {train_dates[0]} to {train_dates[-1]}")
        if val_dates:
            print(f"Validating on {len(val_dates)} days: {val_dates[0]} to {val_dates[-1]}")
        else:
            print("No validation data available!")

        # Get day types for training dates
        train_day_types = self.day_types[self.day_types['Date'].isin(train_dates)]

        # Initialize NHPP models
        self.nhpp_models = {
            'Weekday': NHPP_Model(),
            'Holiday': NHPP_Model()
        }

        # Fit models for each day type
        for day_type in ['Weekday', 'Holiday']:
            print(f"\nFitting NHPP model for {day_type}...")

            # Get dates of this type in training set
            type_dates = train_day_types[train_day_types['DayType'] == day_type]['Date']

            if len(type_dates) == 0:
                print(f"  Warning: No {day_type} data in training set!")
                continue

            # Prepare arrival data
            arrival_data = self._prepare_arrival_data(type_dates)

            # Fit NHPP model
            self.nhpp_models[day_type].fit(arrival_data)

            print(f"  Fitted model with {len(type_dates)} days of data")
            print(f"  Mean arrival rate: {self.nhpp_models[day_type].lambda_t.mean():.2f} passengers/5min")

        return train_dates, val_dates

    def _prepare_arrival_data(self, dates):
        """Prepare arrival count data for given dates"""

        arrival_counts = []

        for date in dates:
            # Filter hall calls for this date
            date_mask = self.hall_calls['Date'] == date
            date_data = self.hall_calls[date_mask]

            # Count arrivals per 5-minute slice
            slices_counts = date_data.groupby('5min_slice').size()

            # Create full array for all 288 slices
            full_counts = np.zeros(288, dtype=np.float32)
            if len(slices_counts) > 0:
                indices = slices_counts.index.astype(int)
                full_counts[indices] = slices_counts.values

            arrival_counts.append(full_counts)

        return np.array(arrival_counts)

    def validate_models(self, val_dates):
        """Validate NHPP models on validation data"""
        print("\n" + "="*60)
        print("MODEL VALIDATION")
        print("="*60)

        if not hasattr(self, 'nhpp_models') or not val_dates:
            print("No models or validation data available!")
            return None

        # Get day types for validation dates
        val_day_types = self.day_types[self.day_types['Date'].isin(val_dates)]

        validation_results = []

        for date in val_dates:
            # Get actual day type
            day_type_row = val_day_types[val_day_types['Date'] == date]
            if len(day_type_row) == 0:
                print(f"Warning: No day type info for {date}")
                continue

            actual_day_type = day_type_row.iloc[0]['DayType']

            # Skip if no model for this day type
            if actual_day_type not in self.nhpp_models:
                continue

            # Get actual arrival counts
            date_mask = self.hall_calls['Date'] == date
            date_data = self.hall_calls[date_mask]
            slices_counts = date_data.groupby('5min_slice').size()

            # Create full array
            actual_counts = np.zeros(288, dtype=np.float32)
            if len(slices_counts) > 0:
                indices = slices_counts.index.astype(int)
                actual_counts[indices] = slices_counts.values

            # Get predictions
            model = self.nhpp_models[actual_day_type]
            predicted_rates = model.lambda_t

            # Calculate prediction errors
            mae = np.mean(np.abs(actual_counts - predicted_rates))
            rmse = np.sqrt(np.mean((actual_counts - predicted_rates)**2))

            # Avoid division by zero for MAPE
            if np.sum(actual_counts) > 0:
                mape = np.sum(np.abs(actual_counts - predicted_rates)) / np.sum(actual_counts) * 100
            else:
                mape = 0

            validation_results.append({
                'Date': date,
                'DayType': actual_day_type,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'TotalActual': actual_counts.sum(),
                'TotalPredicted': predicted_rates.sum()
            })

        if not validation_results:
            print("No validation results!")
            return None

        # Summarize validation results
        val_results_df = pd.DataFrame(validation_results)

        print("\nValidation Results Summary:")
        print("-" * 40)

        for day_type in ['Weekday', 'Holiday']:
            type_results = val_results_df[val_results_df['DayType'] == day_type]
            if len(type_results) > 0:
                print(f"\n{day_type} ({len(type_results)} days):")
                print(f"  Mean Absolute Error (MAE): {type_results['MAE'].mean():.3f}")
                print(f"  Root Mean Square Error (RMSE): {type_results['RMSE'].mean():.3f}")
                print(f"  Mean Absolute Percentage Error (MAPE): {type_results['MAPE'].mean():.2f}%")
                print(f"  Avg Daily Passengers - Actual: {type_results['TotalActual'].mean():.1f}")
                print(f"  Avg Daily Passengers - Predicted: {type_results['TotalPredicted'].mean():.1f}")

        return val_results_df

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        # Create output directory
        os.makedirs('figures', exist_ok=True)

        # Create visualizations
        self._plot_daily_patterns()

        if hasattr(self, 'nhpp_models'):
            self._plot_nhpp_intensities()

        if hasattr(self, 'validation_results') and self.validation_results is not None:
            self._plot_validation_results()

        print("\nVisualizations saved to 'figures/' directory")

    def _plot_daily_patterns(self):
        """Plot daily patterns"""

        if self.hall_calls is None or len(self.hall_calls) == 0:
            print("No data for daily patterns plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Overall hourly distribution
        ax1 = axes[0, 0]
        hourly_counts = self.hall_calls.groupby('Hour').size()
        ax1.bar(hourly_counts.index, hourly_counts.values, alpha=0.7)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Calls')
        ax1.set_title('Overall Hourly Call Distribution')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Day of week distribution
        ax2 = axes[0, 1]
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_counts = self.hall_calls.groupby('DayOfWeek').size()
        ax2.bar(day_counts.index, day_counts.values, alpha=0.7)
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Number of Calls')
        ax2.set_title('Call Distribution by Day of Week')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(day_names)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Date progression (if we have day types)
        ax3 = axes[1, 0]
        if self.day_types is not None:
            # Merge with hall_calls to get daily counts
            daily_counts = self.hall_calls.groupby('Date').size().reset_index()
            daily_counts.columns = ['Date', 'Count']
            daily_counts = pd.merge(daily_counts, self.day_types, on='Date')

            # Sort by date
            daily_counts = daily_counts.sort_values('Date')

            # Plot with different colors for day types
            colors = {'Weekday': 'blue', 'Holiday': 'orange'}
            for day_type, color in colors.items():
                mask = daily_counts['DayType'] == day_type
                if mask.any():
                    ax3.scatter(daily_counts.loc[mask, 'Date'],
                              daily_counts.loc[mask, 'Count'],
                              label=day_type, alpha=0.6, color=color, s=50)

            ax3.set_xlabel('Date')
            ax3.set_ylabel('Daily Calls')
            ax3.set_title('Daily Call Counts with Day Type Classification')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No day type classification available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Daily Call Counts')

        # Plot 4: Floor distribution
        ax4 = axes[1, 1]
        if 'Floor' in self.hall_calls.columns:
            floor_counts = self.hall_calls['Floor'].value_counts().head(20)  # Top 20 floors
            ax4.bar(range(len(floor_counts)), floor_counts.values, alpha=0.7)
            ax4.set_xlabel('Floor Rank')
            ax4.set_ylabel('Number of Calls')
            ax4.set_title('Top 20 Floors by Call Frequency')
            ax4.set_xticks(range(len(floor_counts)))
            ax4.set_xticklabels(floor_counts.index, rotation=45)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No floor data available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Floor Distribution')

        plt.tight_layout()
        plt.savefig('figures/daily_patterns.png', dpi=150, bbox_inches='tight')
        plt.savefig('figures/daily_patterns.pdf', bbox_inches='tight')
        plt.show()

    def _plot_nhpp_intensities(self):
        """Plot NHPP intensity functions"""

        if not hasattr(self, 'nhpp_models'):
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        for idx, (day_type, model) in enumerate(self.nhpp_models.items()):
            if model.lambda_t is None:
                continue

            ax = axes[idx]
            hours = np.arange(288) / 12

            # Plot intensity function
            ax.plot(hours, model.lambda_t, 'b-', linewidth=1.5, label='λ(t)')

            # Highlight peak hours
            ax.axvspan(8, 10, alpha=0.2, color='yellow', label='Morning Peak (8-10)')
            ax.axvspan(12, 14, alpha=0.2, color='orange', label='Lunch (12-14)')
            ax.axvspan(17, 19, alpha=0.2, color='red', label='Evening Peak (17-19)')

            ax.set_xlabel('Time of Day (Hours)')
            ax.set_ylabel('Arrival Rate (passengers/5min)')
            ax.set_title(f'{day_type} NHPP Intensity Function')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 24)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig('figures/nhpp_intensities.png', dpi=150, bbox_inches='tight')
        plt.savefig('figures/nhpp_intensities.pdf', bbox_inches='tight')
        plt.show()

    def _plot_validation_results(self):
        """Plot validation results"""

        if not hasattr(self, 'validation_results'):
            return

        val_df = self.validation_results

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Actual vs Predicted
        ax1 = axes[0, 0]
        colors = {'Weekday': 'blue', 'Holiday': 'orange'}

        for day_type in ['Weekday', 'Holiday']:
            type_data = val_df[val_df['DayType'] == day_type]
            if len(type_data) > 0:
                ax1.scatter(type_data['TotalActual'], type_data['TotalPredicted'],
                          label=day_type, alpha=0.7, s=60, color=colors[day_type])

        # Add perfect prediction line
        max_val = max(val_df['TotalActual'].max(), val_df['TotalPredicted'].max())
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

        ax1.set_xlabel('Actual Total Passengers')
        ax1.set_ylabel('Predicted Total Passengers')
        ax1.set_title('Actual vs Predicted Daily Totals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error metrics by day type
        ax2 = axes[0, 1]
        error_metrics = ['MAE', 'RMSE']
        x_pos = np.arange(len(error_metrics))
        width = 0.35

        for idx, day_type in enumerate(['Weekday', 'Holiday']):
            type_data = val_df[val_df['DayType'] == day_type]
            if len(type_data) > 0:
                means = [type_data[metric].mean() for metric in error_metrics]
                ax2.bar(x_pos + idx*width, means, width, label=day_type, alpha=0.8)

        ax2.set_xlabel('Error Metric')
        ax2.set_ylabel('Value')
        ax2.set_title('Prediction Errors by Day Type')
        ax2.set_xticks(x_pos + width/2)
        ax2.set_xticklabels(error_metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Time series of prediction errors
        ax3 = axes[1, 0]
        val_df_sorted = val_df.sort_values('Date')

        for day_type in ['Weekday', 'Holiday']:
            type_data = val_df_sorted[val_df_sorted['DayType'] == day_type]
            if len(type_data) > 0:
                ax3.plot(type_data['Date'], type_data['MAE'],
                        marker='o', label=f'{day_type} MAE', linewidth=2)

        ax3.set_xlabel('Date')
        ax3.set_ylabel('Mean Absolute Error (MAE)')
        ax3.set_title('Prediction Error Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # Plot 4: Error distribution
        ax4 = axes[1, 1]
        error_data = []
        for _, row in val_df.iterrows():
            error_data.append({'DayType': row['DayType'], 'MAE': row['MAE']})
            error_data.append({'DayType': row['DayType'], 'RMSE': row['RMSE']})

        error_df = pd.DataFrame(error_data)

        # Create box plot
        if len(error_df) > 0:
            error_melted = error_df.melt(id_vars=['DayType'], value_vars=['MAE', 'RMSE'])
            error_melted.columns = ['DayType', 'Metric', 'Value']

            import seaborn as sns
            sns.boxplot(x='Metric', y='Value', hue='DayType', data=error_melted, ax=ax4)

            ax4.set_xlabel('Error Metric')
            ax4.set_ylabel('Error Value')
            ax4.set_title('Error Distribution by Day Type')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'No error data available',
                    ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout()
        plt.savefig('figures/validation_results.png', dpi=150, bbox_inches='tight')
        plt.savefig('figures/validation_results.pdf', bbox_inches='tight')
        plt.show()

    def run_analysis(self):
        """Run the complete analysis pipeline"""

        print("="*60)
        print("MCM PROBLEM B: ELEVATOR TRAFFIC ANALYSIS")
        print("="*60)

        import time
        start_time = time.time()

        # Step 1: Load data
        print("\n[Step 1/6] Loading data...")
        if not self.load_data_safely():
            print("Failed to load data. Exiting.")
            return

        # Step 2: Preprocess data
        print("\n[Step 2/6] Preprocessing data...")
        self.preprocess_data()

        # Step 3: Analyze patterns
        print("\n[Step 3/6] Analyzing traffic patterns...")
        self.analyze_traffic_patterns()

        # Step 4: Fit NHPP models
        print("\n[Step 4/6] Fitting NHPP models...")
        train_dates, val_dates = self.fit_nhpp_models(train_days=20)

        # Step 5: Validate models
        print("\n[Step 5/6] Validating models...")
        if val_dates:
            self.validation_results = self.validate_models(val_dates)
        else:
            print("Skipping validation - no validation data available")

        # Step 6: Create visualizations
        print("\n[Step 6/6] Creating visualizations...")
        self.create_visualizations()

        # Save results
        print("\nSaving analysis results...")
        self.save_results()

        total_time = time.time() - start_time
        print(f"\nAnalysis completed in {total_time:.2f} seconds!")
        print("="*60)

    def save_results(self):
        """Save analysis results to files"""

        # Create output directory
        os.makedirs('results', exist_ok=True)

        # Save day type classification
        if self.day_types is not None:
            self.day_types.to_csv('results/day_type_classification.csv', index=False)
            print("Day type classification saved to 'results/day_type_classification.csv'")

        # Save validation results
        if hasattr(self, 'validation_results') and self.validation_results is not None:
            self.validation_results.to_csv('results/validation_results.csv', index=False)
            print("Validation results saved to 'results/validation_results.csv'")

        # Save NHPP model parameters
        if hasattr(self, 'nhpp_models'):
            nhpp_params = {}
            for day_type, model in self.nhpp_models.items():
                if model.lambda_t is not None:
                    nhpp_params[day_type] = model.lambda_t

            np.savez_compressed('results/nhpp_parameters.npz', **nhpp_params)
            print("NHPP model parameters saved to 'results/nhpp_parameters.npz'")

        print("All results saved to 'results/' directory")


class NHPP_Model:
    """Non-Homogeneous Poisson Process Model"""

    def __init__(self):
        self.lambda_t = None
        self.fitted = False

    def fit(self, arrival_data):
        """Fit NHPP model to arrival data"""

        if len(arrival_data) == 0:
            print("  Warning: No arrival data to fit!")
            return

        # Estimate λ(t) as the mean arrival rate at each time slice
        self.lambda_t = np.mean(arrival_data, axis=0).astype(np.float32)

        # Apply smoothing with moving average
        window_size = 3  # 15-minute window
        if window_size > 0 and len(self.lambda_t) > window_size:
            kernel = np.ones(window_size) / window_size
            self.lambda_t = np.convolve(self.lambda_t, kernel, mode='same')

        self.fitted = True

    def predict(self, time_slice):
        """Predict arrival rate for a given time slice"""
        if not self.fitted:
            raise ValueError("Model not fitted yet!")

        if time_slice < 0 or time_slice >= 288:
            raise ValueError(f"Time slice must be between 0 and 287, got {time_slice}")

        return self.lambda_t[time_slice]


# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ElevatorDataAnalyzer(data_folder='./data')

    # Run analysis
    analyzer.run_analysis()

    print("\nAnalysis complete! Check the 'figures/' and 'results/' directories.")