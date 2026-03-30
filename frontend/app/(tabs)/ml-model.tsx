import { ScrollView, Text, View, Pressable } from "react-native";
import { useState } from "react";
import { ScreenContainer } from "@/components/screen-container";
import { useColors } from "@/hooks/use-colors";
import * as Haptics from "expo-haptics";

interface FeatureImportance {
  name: string;
  importance: number;
}

interface ModelMetric {
  label: string;
  value: number;
  unit: string;
}

const FEATURE_IMPORTANCE: FeatureImportance[] = [
  { name: "Venue", importance: 0.18 },
  { name: "Toss Decision", importance: 0.15 },
  { name: "Current RRR", importance: 0.14 },
  { name: "Wickets Lost", importance: 0.12 },
  { name: "Momentum (18B)", importance: 0.11 },
  { name: "Player Form", importance: 0.1 },
  { name: "Weather", importance: 0.08 },
  { name: "Historical H2H", importance: 0.07 },
  { name: "Pitch Report", importance: 0.05 },
];

const MODEL_METRICS: ModelMetric[] = [
  { label: "Accuracy", value: 78.4, unit: "%" },
  { label: "Precision", value: 81.2, unit: "%" },
  { label: "Recall", value: 76.8, unit: "%" },
  { label: "F1-Score", value: 0.79, unit: "" },
];

const PREDICTION_BREAKDOWN = {
  xgboost: 65,
  lstm: 72,
  ensemble: 72,
};

function FeatureBar({ feature, importance }: { feature: string; importance: number }) {
  const percentage = importance * 100;

  return (
    <View className="mb-3">
      <View className="flex-row justify-between items-center mb-1">
        <Text className="text-xs font-semibold text-foreground flex-1">
          {feature}
        </Text>
        <Text className="text-xs text-muted">{percentage.toFixed(1)}%</Text>
      </View>
      <View className="h-2 bg-background rounded-full overflow-hidden">
        <View
          className="h-full bg-primary rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </View>
    </View>
  );
}

function MetricBox({ metric }: { metric: ModelMetric }) {
  return (
    <View className="bg-surface rounded-lg p-3 border border-border flex-1 items-center">
      <Text className="text-xs text-muted mb-1">{metric.label}</Text>
      <Text className="text-lg font-bold text-primary">
        {metric.value}
        <Text className="text-xs text-muted">{metric.unit}</Text>
      </Text>
    </View>
  );
}

function CodeViewer({ title, code }: { title: string; code: string }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <View className="bg-surface rounded-lg border border-border overflow-hidden mb-3">
      <Pressable
        onPress={() => {
          setExpanded(!expanded);
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        }}
        className="p-4 flex-row justify-between items-center"
      >
        <Text className="text-sm font-semibold text-foreground">{title}</Text>
        <Text className="text-lg">{expanded ? "▼" : "▶"}</Text>
      </Pressable>
      {expanded && (
        <View className="bg-background p-3 border-t border-border">
          <Text className="text-xs font-mono text-muted leading-relaxed">
            {code}
          </Text>
        </View>
      )}
    </View>
  );
}

export default function MLModelScreen() {
  const colors = useColors();

  const architectureCode = `# Hybrid Ensemble Model
import xgboost as xgb
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.models import Model

# XGBoost for static features
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05
)

# LSTM/GRU for time-series (18 balls)
lstm_input = Input(shape=(18, 8))
lstm_out = LSTM(64, return_sequences=True)(lstm_input)
lstm_out = GRU(32)(lstm_out)
lstm_out = Dense(16, activation='relu')(lstm_out)

# Ensemble
ensemble = Concatenate()([xgb_out, lstm_out])
output = Dense(1, activation='sigmoid')(ensemble)`;

  const trainingCode = `# Data Normalization (Cricsheet)
from sklearn.preprocessing import StandardScaler

# Load ball-by-ball data
df = load_cricsheet_data()

# Feature engineering
df['runs_rate'] = df['runs'] / df['overs']
df['wicket_loss_rate'] = df['wickets'] / df['overs']
df['momentum'] = df['last_18_balls'].rolling(18).mean()

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split: 70% train, 15% val, 15% test
X_train, X_test = train_test_split(X_scaled, 0.85)`;

  return (
    <ScreenContainer className="p-0">
      <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
        {/* Header */}
        <View className="px-4 pt-4 pb-2">
          <Text className="text-2xl font-bold text-foreground">
            ML Model
          </Text>
          <Text className="text-sm text-muted mt-1">
            Hybrid XGBoost + LSTM/GRU ensemble
          </Text>
        </View>

        {/* Model Architecture */}
        <View className="px-4 py-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Architecture
          </Text>
          <View className="bg-surface rounded-lg p-4 border border-border mb-4">
            <View className="gap-3">
              {/* Input Layer */}
              <View className="items-center">
                <View className="bg-primary/20 border border-primary rounded-lg px-4 py-2 w-full">
                  <Text className="text-xs font-semibold text-foreground text-center">
                    Input Features
                  </Text>
                  <Text className="text-xs text-muted text-center">
                    Venue, Toss, XI, RRR, CRR, Momentum
                  </Text>
                </View>
              </View>

              {/* Split */}
              <View className="flex-row gap-2">
                <View className="flex-1 items-center">
                  <View className="h-8 w-px bg-border" />
                </View>
                <View className="flex-1 items-center">
                  <View className="h-8 w-px bg-border" />
                </View>
              </View>

              {/* XGBoost and LSTM */}
              <View className="flex-row gap-2">
                <View className="flex-1 bg-success/10 border border-success rounded-lg px-3 py-2">
                  <Text className="text-xs font-semibold text-foreground text-center">
                    XGBoost
                  </Text>
                  <Text className="text-xs text-muted text-center">
                    Static Features
                  </Text>
                </View>
                <View className="flex-1 bg-warning/10 border border-warning rounded-lg px-3 py-2">
                  <Text className="text-xs font-semibold text-foreground text-center">
                    LSTM/GRU
                  </Text>
                  <Text className="text-xs text-muted text-center">
                    Time-Series (18B)
                  </Text>
                </View>
              </View>

              {/* Merge */}
              <View className="flex-row gap-2">
                <View className="flex-1 items-center">
                  <View className="h-8 w-px bg-border" />
                </View>
                <View className="flex-1 items-center">
                  <View className="h-8 w-px bg-border" />
                </View>
              </View>

              {/* Output */}
              <View className="items-center">
                <View className="bg-primary/20 border border-primary rounded-lg px-4 py-2 w-full">
                  <Text className="text-xs font-semibold text-foreground text-center">
                    Ensemble Output
                  </Text>
                  <Text className="text-xs text-muted text-center">
                    Win Probability (0-100%)
                  </Text>
                </View>
              </View>
            </View>
          </View>
        </View>

        {/* Performance Metrics */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Performance Metrics
          </Text>
          <View className="flex-row gap-2">
            {MODEL_METRICS.slice(0, 2).map((metric, idx) => (
              <MetricBox key={idx} metric={metric} />
            ))}
          </View>
          <View className="flex-row gap-2 mt-2">
            {MODEL_METRICS.slice(2, 4).map((metric, idx) => (
              <MetricBox key={idx} metric={metric} />
            ))}
          </View>
        </View>

        {/* Prediction Breakdown */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Current Prediction Breakdown
          </Text>
          <View className="bg-surface rounded-lg p-4 border border-border">
            <View className="mb-4">
              <View className="flex-row justify-between items-center mb-2">
                <Text className="text-sm font-semibold text-foreground">
                  XGBoost
                </Text>
                <Text className="text-lg font-bold text-success">
                  {PREDICTION_BREAKDOWN.xgboost}%
                </Text>
              </View>
              <View className="h-2 bg-background rounded-full overflow-hidden">
                <View
                  className="h-full bg-success rounded-full"
                  style={{ width: `${PREDICTION_BREAKDOWN.xgboost}%` }}
                />
              </View>
            </View>

            <View className="mb-4">
              <View className="flex-row justify-between items-center mb-2">
                <Text className="text-sm font-semibold text-foreground">
                  LSTM/GRU
                </Text>
                <Text className="text-lg font-bold text-warning">
                  {PREDICTION_BREAKDOWN.lstm}%
                </Text>
              </View>
              <View className="h-2 bg-background rounded-full overflow-hidden">
                <View
                  className="h-full bg-warning rounded-full"
                  style={{ width: `${PREDICTION_BREAKDOWN.lstm}%` }}
                />
              </View>
            </View>

            <View className="pt-3 border-t border-border">
              <View className="flex-row justify-between items-center">
                <Text className="text-sm font-semibold text-foreground">
                  Ensemble Final
                </Text>
                <Text className="text-lg font-bold text-primary">
                  {PREDICTION_BREAKDOWN.ensemble}%
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* Feature Importance */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Feature Importance
          </Text>
          {FEATURE_IMPORTANCE.map((feature, idx) => (
            <FeatureBar
              key={idx}
              feature={feature.name}
              importance={feature.importance}
            />
          ))}
        </View>

        {/* Code Viewers */}
        <View className="px-4 pb-4">
          <Text className="text-lg font-semibold text-foreground mb-3">
            Implementation
          </Text>
          <CodeViewer title="Hybrid Architecture" code={architectureCode} />
          <CodeViewer title="Data Normalization" code={trainingCode} />
        </View>
      </ScrollView>
    </ScreenContainer>
  );
}
